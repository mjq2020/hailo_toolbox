"""
Unified Data Interface Pipeline Example

This example demonstrates how to use the unified data interface system
to build complete inference pipelines with different components and configurations.
"""

import numpy as np
from typing import Dict, Any
import logging
import tempfile
import os

from .interface import (
    Pipeline, PipelineConfig, DataContainer, Metadata,
    DataType, ProcessingStage, create_image_container
)
from .adapters import (
    ImageSourceAdapter, PreprocessorAdapter, InferenceAdapter,
    PostprocessorAdapter, VisualizationAdapter, OutputAdapter
)
from .base import PostprocessConfig
from .preprocessor import PreprocessConfig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockInferenceEngine:
    """
    Mock inference engine for demonstration purposes.
    
    This simulates a real inference engine like Hailo or ONNX Runtime.
    """
    
    def __init__(self, task_type: str = "detection", num_classes: int = 80):
        """
        Initialize mock inference engine.
        
        Args:
            task_type: Type of task ('detection', 'segmentation', 'keypoint')
            num_classes: Number of classes for classification tasks
        """
        self.task_type = task_type
        self.num_classes = num_classes
        self.is_warmed_up = False
        
        logger.info(f"Initialized MockInferenceEngine for {task_type}")
    
    def warm_up(self, input_shape):
        """Warm up the inference engine."""
        logger.info(f"Warming up inference engine with input shape: {input_shape}")
        self.is_warmed_up = True
    
    def infer(self, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on input tensor.
        
        Args:
            input_tensor: Input tensor with shape (batch, channels, height, width)
            
        Returns:
            Dictionary of output tensors
        """
        if not self.is_warmed_up:
            logger.warning("Inference engine not warmed up")
        
        batch_size = input_tensor.shape[0] if len(input_tensor.shape) == 4 else 1
        
        if self.task_type == "detection":
            # Mock detection output: [x, y, w, h, class_0, class_1, ...]
            num_detections = 10
            num_features = 4 + self.num_classes
            
            # Generate realistic detection data
            np.random.seed(42)  # For reproducible results
            detection_output = np.random.rand(batch_size, num_detections, num_features)
            
            # Set some detections to have high confidence
            detection_output[:, :5, 4:6] = 0.8  # High confidence for first 5 detections
            detection_output[:, 5:, 4:] = 0.1   # Low confidence for rest
            
            # Set reasonable bounding box coordinates (normalized to 0-1)
            detection_output[:, :, 0] = np.random.uniform(0.1, 0.9, (batch_size, num_detections))  # x
            detection_output[:, :, 1] = np.random.uniform(0.1, 0.9, (batch_size, num_detections))  # y
            detection_output[:, :, 2] = np.random.uniform(0.05, 0.3, (batch_size, num_detections))  # w
            detection_output[:, :, 3] = np.random.uniform(0.05, 0.3, (batch_size, num_detections))  # h
            
            return {"output": detection_output}
            
        elif self.task_type == "segmentation":
            # Mock segmentation output: detection + mask prototypes
            num_detections = 5
            num_mask_coeffs = 32
            num_features = 4 + self.num_classes + num_mask_coeffs
            
            np.random.seed(42)
            detection_output = np.random.rand(batch_size, num_detections, num_features)
            detection_output[:, :, 4:6] = 0.8  # High class scores
            
            # Mock mask prototypes
            proto_h, proto_w = 160, 160
            mask_prototypes = np.random.randn(batch_size, num_mask_coeffs, proto_h, proto_w)
            
            return {
                "output0": detection_output,
                "output1": mask_prototypes
            }
            
        elif self.task_type == "keypoint":
            # Mock keypoint output: [x, y, w, h, person_conf, kp1_x, kp1_y, kp1_vis, ...]
            num_persons = 3
            num_keypoints = 17
            num_features = 5 + num_keypoints * 3
            
            np.random.seed(42)
            keypoint_output = np.random.rand(batch_size, num_persons, num_features)
            
            # Set person confidence
            keypoint_output[:, :, 4] = [0.9, 0.8, 0.7]
            
            # Set reasonable keypoint coordinates and visibility
            for i in range(num_persons):
                kp_start = 5
                for j in range(num_keypoints):
                    kp_idx = kp_start + j * 3
                    keypoint_output[:, i, kp_idx:kp_idx+2] = np.random.rand(2) * 640  # x, y
                    keypoint_output[:, i, kp_idx+2] = np.random.rand()  # visibility
            
            return {"output": keypoint_output}
        
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")


def create_mock_image(width: int = 640, height: int = 640) -> np.ndarray:
    """
    Create a mock image for testing.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Mock RGB image
    """
    # Create a colorful test image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some patterns
    image[:height//2, :width//2] = [255, 0, 0]    # Red quadrant
    image[:height//2, width//2:] = [0, 255, 0]    # Green quadrant
    image[height//2:, :width//2] = [0, 0, 255]    # Blue quadrant
    image[height//2:, width//2:] = [255, 255, 0]  # Yellow quadrant
    
    # Add some noise for realism
    noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


def example_detection_pipeline():
    """
    Example of a complete detection pipeline using the unified interface.
    """
    print("\n" + "="*60)
    print("DETECTION PIPELINE EXAMPLE")
    print("="*60)
    
    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        temp_image_path = tmp_file.name
    
    try:
        # Save mock image
        import cv2
        mock_image = create_mock_image()
        cv2.imwrite(temp_image_path, cv2.cvtColor(mock_image, cv2.COLOR_RGB2BGR))
        
        # Create pipeline configuration
        config = PipelineConfig(
            source_config={'path': temp_image_path, 'type': 'file'},
            preprocess_config={'target_size': (640, 640), 'normalize': True},
            inference_config={'task_type': 'detection'},
            postprocess_config={'num_classes': 80, 'det_conf_threshold': 0.25},
            visualization_config={},
            output_config={'path': temp_image_path.replace('.jpg', '_output.jpg'), 'format': 'image'}
        )
        
        # Create pipeline components
        source = ImageSourceAdapter(temp_image_path, "file")
        preprocessor = PreprocessorAdapter(PreprocessConfig(
            target_size=(640, 640),
            normalize=True
        ))
        inference_engine = MockInferenceEngine("detection", 80)
        inference = InferenceAdapter(inference_engine, {
            'name': 'YOLOv8-Detection',
            'version': '1.0',
            'task': 'detection'
        })
        postprocessor = PostprocessorAdapter("detection", PostprocessConfig(
            num_classes=80,
            det_conf_threshold=0.25,
            nms_iou_threshold=0.5
        ))
        visualization = VisualizationAdapter()
        output = OutputAdapter(temp_image_path.replace('.jpg', '_output.jpg'), "image")
        
        # Build pipeline
        pipeline = Pipeline(config)
        pipeline.add_component(source) \
               .add_component(preprocessor) \
               .add_component(inference) \
               .add_component(postprocessor) \
               .add_component(visualization) \
               .add_component(output)
        
        # Initialize and run pipeline
        pipeline.initialize()
        result = pipeline.process_single()
        
        # Print results
        print(f"Pipeline completed successfully!")
        print(f"Processing stages: {[stage.value for stage in pipeline.components.keys()]}")
        print(f"Final data type: {result.metadata.data_type.value}")
        print(f"Processing times: {result.metadata.processing_times}")
        
        if result.postprocessed_results is not None:
            detection_results = result.postprocessed_results
            print(f"Number of detections: {len(detection_results)}")
            if len(detection_results) > 0:
                print(f"Detection boxes shape: {detection_results.boxes.shape}")
                print(f"Confidence scores: {detection_results.scores[:5]}")  # First 5 scores
                print(f"Class IDs: {detection_results.class_ids[:5]}")  # First 5 class IDs
        
        # Get statistics
        stats = pipeline.get_statistics()
        print(f"Pipeline statistics: {stats}")
        
        # Cleanup
        pipeline.cleanup()
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
        output_path = temp_image_path.replace('.jpg', '_output.jpg')
        if os.path.exists(output_path):
            os.unlink(output_path)


def example_segmentation_pipeline():
    """
    Example of a complete segmentation pipeline using the unified interface.
    """
    print("\n" + "="*60)
    print("SEGMENTATION PIPELINE EXAMPLE")
    print("="*60)
    
    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        temp_image_path = tmp_file.name
    
    try:
        # Save mock image
        import cv2
        mock_image = create_mock_image()
        cv2.imwrite(temp_image_path, cv2.cvtColor(mock_image, cv2.COLOR_RGB2BGR))
        
        # Create pipeline configuration
        config = PipelineConfig(
            enable_profiling=True,
            continue_on_error=False
        )
        
        # Create pipeline components
        source = ImageSourceAdapter(temp_image_path, "file")
        preprocessor = PreprocessorAdapter(PreprocessConfig(
            target_size=(640, 640),
            normalize=True
        ))
        inference_engine = MockInferenceEngine("segmentation", 80)
        inference = InferenceAdapter(inference_engine, {
            'name': 'YOLOv8-Segmentation',
            'version': '1.0',
            'task': 'segmentation'
        })
        postprocessor = PostprocessorAdapter("segmentation", PostprocessConfig(
            num_classes=80,
            seg_conf_threshold=0.25,
            seg_mask_threshold=0.5,
            nms_iou_threshold=0.5
        ))
        visualization = VisualizationAdapter()
        
        # Build pipeline
        pipeline = Pipeline(config)
        pipeline.add_component(source) \
               .add_component(preprocessor) \
               .add_component(inference) \
               .add_component(postprocessor) \
               .add_component(visualization)
        
        # Initialize and run pipeline
        pipeline.initialize()
        result = pipeline.process_single()
        
        # Print results
        print(f"Pipeline completed successfully!")
        print(f"Final data type: {result.metadata.data_type.value}")
        
        if result.postprocessed_results is not None:
            seg_results = result.postprocessed_results
            print(f"Number of instances: {len(seg_results)}")
            if len(seg_results) > 0:
                print(f"Mask shape: {seg_results.masks.shape}")
                print(f"Detection boxes shape: {seg_results.boxes.shape}")
                print(f"Confidence scores: {seg_results.scores[:3]}")  # First 3 scores
                print(f"Class IDs: {seg_results.class_ids[:3]}")  # First 3 class IDs
                
                # Calculate mask coverage
                for i in range(min(3, len(seg_results))):
                    mask_coverage = np.sum(seg_results.masks[i]) / seg_results.masks[i].size * 100
                    print(f"Mask {i} coverage: {mask_coverage:.2f}%")
        
        # Cleanup
        pipeline.cleanup()
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.unlink(temp_image_path)


def example_keypoint_pipeline():
    """
    Example of a complete keypoint detection pipeline using the unified interface.
    """
    print("\n" + "="*60)
    print("KEYPOINT DETECTION PIPELINE EXAMPLE")
    print("="*60)
    
    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        temp_image_path = tmp_file.name
    
    try:
        # Save mock image
        import cv2
        mock_image = create_mock_image()
        cv2.imwrite(temp_image_path, cv2.cvtColor(mock_image, cv2.COLOR_RGB2BGR))
        
        # Create pipeline configuration
        config = PipelineConfig()
        
        # Create pipeline components
        source = ImageSourceAdapter(temp_image_path, "file")
        preprocessor = PreprocessorAdapter(PreprocessConfig(
            target_size=(640, 640),
            normalize=True
        ))
        inference_engine = MockInferenceEngine("keypoint", 17)
        inference = InferenceAdapter(inference_engine, {
            'name': 'YOLOv8-Pose',
            'version': '1.0',
            'task': 'keypoint'
        })
        postprocessor = PostprocessorAdapter("keypoint", PostprocessConfig(
            num_keypoints=17,
            kp_conf_threshold=0.25,
            kp_visibility_threshold=0.5,
            nms_iou_threshold=0.5
        ))
        visualization = VisualizationAdapter()
        
        # Build pipeline
        pipeline = Pipeline(config)
        pipeline.add_component(source) \
               .add_component(preprocessor) \
               .add_component(inference) \
               .add_component(postprocessor) \
               .add_component(visualization)
        
        # Initialize and run pipeline
        pipeline.initialize()
        result = pipeline.process_single()
        
        # Print results
        print(f"Pipeline completed successfully!")
        print(f"Final data type: {result.metadata.data_type.value}")
        
        if result.postprocessed_results is not None:
            kp_results = result.postprocessed_results
            print(f"Number of persons detected: {len(kp_results)}")
            if len(kp_results) > 0:
                print(f"Keypoints shape: {kp_results.keypoints.shape}")
                print(f"Person boxes shape: {kp_results.boxes.shape}")
                print(f"Person confidence scores: {kp_results.scores}")
                
                # Count visible keypoints for each person
                for i in range(len(kp_results)):
                    visible_kps = np.sum(kp_results.keypoints[i, :, 2] > 0.5)
                    print(f"Person {i}: {visible_kps} visible keypoints")
        
        # Cleanup
        pipeline.cleanup()
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.unlink(temp_image_path)


def example_batch_processing():
    """
    Example of batch processing using the unified interface.
    """
    print("\n" + "="*60)
    print("BATCH PROCESSING EXAMPLE")
    print("="*60)
    
    # Create multiple mock images
    temp_files = []
    try:
        import cv2
        
        # Create 3 different mock images
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=f'_batch_{i}.jpg', delete=False) as tmp_file:
                temp_image_path = tmp_file.name
                temp_files.append(temp_image_path)
                
                # Create slightly different images
                mock_image = create_mock_image()
                # Add different patterns for each image
                if i == 1:
                    mock_image = np.roll(mock_image, 100, axis=0)  # Shift vertically
                elif i == 2:
                    mock_image = np.roll(mock_image, 100, axis=1)  # Shift horizontally
                
                cv2.imwrite(temp_image_path, cv2.cvtColor(mock_image, cv2.COLOR_RGB2BGR))
        
        # Create pipeline configuration
        config = PipelineConfig(batch_size=3)
        
        # Create pipeline components (reusable for all images)
        preprocessor = PreprocessorAdapter(PreprocessConfig(
            target_size=(640, 640),
            normalize=True
        ))
        inference_engine = MockInferenceEngine("detection", 80)
        inference = InferenceAdapter(inference_engine, {
            'name': 'YOLOv8-Detection-Batch',
            'version': '1.0',
            'task': 'detection'
        })
        postprocessor = PostprocessorAdapter("detection", PostprocessConfig(
            num_classes=80,
            det_conf_threshold=0.25
        ))
        
        # Process each image
        results = []
        for i, image_path in enumerate(temp_files):
            print(f"\nProcessing image {i+1}/{len(temp_files)}: {os.path.basename(image_path)}")
            
            # Create source for this image
            source = ImageSourceAdapter(image_path, "file")
            
            # Build pipeline for this image
            pipeline = Pipeline(config)
            pipeline.add_component(source) \
                   .add_component(preprocessor) \
                   .add_component(inference) \
                   .add_component(postprocessor)
            
            # Initialize and run
            pipeline.initialize()
            result = pipeline.process_single()
            results.append(result)
            
            # Print individual results
            if result.postprocessed_results is not None:
                det_results = result.postprocessed_results
                print(f"  Detections: {len(det_results)}")
                if len(det_results) > 0:
                    print(f"  Top confidence: {np.max(det_results.scores):.3f}")
            
            pipeline.cleanup()
        
        # Print batch summary
        print(f"\nBatch processing completed!")
        print(f"Total images processed: {len(results)}")
        total_detections = sum(len(r.postprocessed_results) if r.postprocessed_results else 0 for r in results)
        print(f"Total detections across all images: {total_detections}")
        
        # Calculate average processing times
        avg_times = {}
        for result in results:
            for stage, time_val in result.metadata.processing_times.items():
                if stage not in avg_times:
                    avg_times[stage] = []
                avg_times[stage].append(time_val)
        
        print("Average processing times:")
        for stage, times in avg_times.items():
            print(f"  {stage}: {np.mean(times):.4f}s ± {np.std(times):.4f}s")
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


def example_error_handling():
    """
    Example demonstrating error handling in the unified interface.
    """
    print("\n" + "="*60)
    print("ERROR HANDLING EXAMPLE")
    print("="*60)
    
    # Test with non-existent image file
    try:
        config = PipelineConfig(continue_on_error=True)
        source = ImageSourceAdapter("/non/existent/path.jpg", "file")
        
        pipeline = Pipeline(config)
        pipeline.add_component(source)
        
        pipeline.initialize()
        result = pipeline.process_single()
        
        print("Error handling test completed (should have handled gracefully)")
        
    except Exception as e:
        print(f"Expected error caught: {e}")
    
    # Test with invalid configuration
    try:
        invalid_config = PostprocessConfig(det_conf_threshold=2.0)  # Invalid threshold
        print("This should not print - invalid config should raise error")
    except ValueError as e:
        print(f"Configuration validation error (expected): {e}")


def main():
    """
    Run all pipeline examples.
    """
    print("UNIFIED DATA INTERFACE PIPELINE EXAMPLES")
    print("="*80)
    
    try:
        # Run individual pipeline examples
        example_detection_pipeline()
        example_segmentation_pipeline()
        example_keypoint_pipeline()
        
        # Run batch processing example
        example_batch_processing()
        
        # Run error handling example
        example_error_handling()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        raise


if __name__ == "__main__":
    main() 