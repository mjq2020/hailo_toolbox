"""
Adapter Module for Unified Data Interface System

This module provides adapter classes that wrap existing components to work
with the new unified data interface system. This allows for gradual migration
and backward compatibility while maintaining the benefits of the new interface.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import cv2
import logging

from .interface import (
    DataContainer, Metadata, DataType, ProcessingStage,
    SourceInterface, PreprocessorInterface, InferenceInterface,
    PostprocessorInterface, VisualizationInterface, OutputInterface,
    create_image_container
)
from .base import PostprocessConfig
from .preprocessor import ImagePreprocessor, PreprocessConfig
from .postprocessor_det import YOLOv8DetPostprocessor
from .postprocessor_seg import YOLOv8SegPostprocessor
from .postprocessor_kp import YOLOv8KpPostprocessor


logger = logging.getLogger(__name__)


class ImageSourceAdapter(SourceInterface):
    """
    Adapter for image sources (files, cameras, etc.).
    
    This adapter wraps various image sources and provides a unified interface
    for the pipeline system.
    """
    
    def __init__(self, source_path: str, source_type: str = "file"):
        """
        Initialize the image source adapter.
        
        Args:
            source_path: Path to image file or camera device
            source_type: Type of source ('file', 'camera', 'stream')
        """
        self.source_path = source_path
        self.source_type = source_type
        self.is_camera = source_type == "camera"
        self.cap = None
        
        if self.is_camera:
            self.cap = cv2.VideoCapture(int(source_path) if source_path.isdigit() else source_path)
        
        logger.info(f"Initialized ImageSourceAdapter for {source_type}: {source_path}")
    
    def get_stage(self) -> ProcessingStage:
        """Get the processing stage."""
        return ProcessingStage.SOURCE
    
    def is_available(self) -> bool:
        """Check if the source is available."""
        if self.is_camera:
            return self.cap is not None and self.cap.isOpened()
        else:
            from pathlib import Path
            return Path(self.source_path).exists()
    
    def get_properties(self) -> Dict[str, Any]:
        """Get source properties."""
        properties = {
            'source_path': self.source_path,
            'source_type': self.source_type,
            'is_camera': self.is_camera
        }
        
        if self.is_camera and self.cap is not None:
            properties.update({
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS)
            })
        
        return properties
    
    def process(self, input_data: DataContainer[None]) -> DataContainer[np.ndarray]:
        """
        Process source data and return image container.
        
        Args:
            input_data: Input container (unused for sources)
            
        Returns:
            Image data container
        """
        if self.is_camera:
            if not self.is_available():
                raise RuntimeError("Camera source not available")
            
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from camera")
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        else:
            # Load image from file
            if not self.is_available():
                raise FileNotFoundError(f"Image file not found: {self.source_path}")
            
            frame = cv2.imread(self.source_path)
            if frame is None:
                raise ValueError(f"Failed to load image: {self.source_path}")
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create metadata
        metadata = Metadata(
            source_path=self.source_path,
            source_type=self.source_type,
            data_type=DataType.IMAGE,
            original_shape=frame.shape,
            current_shape=frame.shape,
            processing_stage=ProcessingStage.SOURCE
        )
        
        return DataContainer(data=frame, metadata=metadata)
    
    def cleanup(self):
        """Cleanup resources."""
        if self.cap is not None:
            self.cap.release()


class PreprocessorAdapter(PreprocessorInterface):
    """
    Adapter for image preprocessors.
    
    This adapter wraps the existing ImagePreprocessor to work with
    the unified data interface.
    """
    
    def __init__(self, config: PreprocessConfig = None):
        """
        Initialize the preprocessor adapter.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessConfig()
        self.preprocessor = ImagePreprocessor(self.config)
        
        logger.info("Initialized PreprocessorAdapter")
    
    def get_stage(self) -> ProcessingStage:
        """Get the processing stage."""
        return ProcessingStage.PREPROCESSING
    
    def get_input_spec(self) -> Dict[str, Any]:
        """Get input specification requirements."""
        return {
            'data_type': 'image',
            'format': 'numpy.ndarray',
            'channels': 3,
            'dtype': 'uint8'
        }
    
    def get_output_spec(self) -> Dict[str, Any]:
        """Get output specification for this adapter."""
        # Determine output shape based on configuration
        if self.config.target_size is not None:
            width, height = self.config.target_size  # target_size is (width, height)
        else:
            height, width = 640, 640  # Default size
        
        return {
            'data_type': DataType.TENSOR,
            'output_stage': ProcessingStage.PREPROCESSING,
            'shape': (1, 3, height, width),
            'dtype': 'float32'
        }
    
    def process(self, input_data: DataContainer[np.ndarray]) -> DataContainer[np.ndarray]:
        """
        Process image data through preprocessing.
        
        Args:
            input_data: Input image container
            
        Returns:
            Preprocessed tensor container
        """
        # Extract image from container
        image = input_data.data
        
        # Apply preprocessing
        preprocessed_tensor = self.preprocessor(image)
        
        # Update metadata
        new_metadata = input_data.metadata.copy()
        new_metadata.update_stage(ProcessingStage.PREPROCESSING)
        new_metadata.current_shape = preprocessed_tensor.shape
        new_metadata.data_type = DataType.TENSOR
        
        # Create new container
        result_container = DataContainer(
            data=preprocessed_tensor,
            metadata=new_metadata,
            raw_data=input_data.data,
            preprocessed_data=preprocessed_tensor
        )
        
        return result_container


class InferenceAdapter(InferenceInterface):
    """
    Adapter for inference engines.
    
    This adapter provides a unified interface for different inference engines
    (Hailo, ONNX, TensorRT, etc.).
    """
    
    def __init__(self, inference_engine, model_info: Dict[str, Any] = None):
        """
        Initialize the inference adapter.
        
        Args:
            inference_engine: The actual inference engine instance
            model_info: Model information dictionary
        """
        self.inference_engine = inference_engine
        self.model_info = model_info or {}
        
        logger.info(f"Initialized InferenceAdapter for {type(inference_engine).__name__}")
    
    def get_stage(self) -> ProcessingStage:
        """Get the processing stage."""
        return ProcessingStage.INFERENCE
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.model_info
    
    def warm_up(self, input_shape: Tuple[int, ...]) -> None:
        """Warm up the inference engine."""
        if hasattr(self.inference_engine, 'warm_up'):
            self.inference_engine.warm_up(input_shape)
        else:
            # Create dummy input for warmup
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            try:
                self.inference_engine.infer(dummy_input)
                logger.info("Inference engine warmed up successfully")
            except Exception as e:
                logger.warning(f"Warmup failed: {e}")
    
    def process(self, input_data: DataContainer[np.ndarray]) -> DataContainer[Dict[str, np.ndarray]]:
        """
        Process tensor data through inference.
        
        Args:
            input_data: Input tensor container
            
        Returns:
            Inference outputs container
        """
        # Extract preprocessed tensor
        tensor = input_data.preprocessed_data if input_data.preprocessed_data is not None else input_data.data
        
        # Run inference
        if hasattr(self.inference_engine, 'infer'):
            outputs = self.inference_engine.infer(tensor)
        elif hasattr(self.inference_engine, '__call__'):
            outputs = self.inference_engine(tensor)
        else:
            raise ValueError("Inference engine must have 'infer' or '__call__' method")
        
        # Ensure outputs is a dictionary
        if not isinstance(outputs, dict):
            if isinstance(outputs, (list, tuple)):
                outputs = {f"output{i}": out for i, out in enumerate(outputs)}
            else:
                outputs = {"output": outputs}
        
        # Update metadata
        new_metadata = input_data.metadata.copy()
        new_metadata.update_stage(ProcessingStage.INFERENCE)
        new_metadata.model_name = self.model_info.get('name', 'unknown')
        new_metadata.model_version = self.model_info.get('version', 'unknown')
        
        # Create new container
        result_container = DataContainer(
            data=outputs,
            metadata=new_metadata,
            raw_data=input_data.raw_data,
            preprocessed_data=input_data.preprocessed_data,
            inference_outputs=outputs
        )
        
        return result_container


class PostprocessorAdapter(PostprocessorInterface):
    """
    Adapter for postprocessors.
    
    This adapter wraps YOLOv8 postprocessors to work with the unified interface.
    """
    
    def __init__(self, task_type: str, config: PostprocessConfig = None):
        """
        Initialize the postprocessor adapter.
        
        Args:
            task_type: Type of task ('det', 'seg', 'kp')
            config: Postprocessing configuration
        """
        self.task_type = task_type.lower()
        self.config = config or PostprocessConfig()
        
        # Create appropriate postprocessor
        if self.task_type in ['det', 'detection']:
            self.postprocessor = YOLOv8DetPostprocessor(self.config)
        elif self.task_type in ['seg', 'segmentation']:
            self.postprocessor = YOLOv8SegPostprocessor(self.config)
        elif self.task_type in ['kp', 'keypoint', 'pose']:
            self.postprocessor = YOLOv8KpPostprocessor(self.config)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        logger.info(f"Initialized PostprocessorAdapter for {task_type}")
    
    def get_stage(self) -> ProcessingStage:
        """Get the processing stage."""
        return ProcessingStage.POSTPROCESSING
    
    def get_supported_tasks(self) -> List[str]:
        """Get list of supported task types."""
        return [self.task_type]
    
    def process(self, input_data: DataContainer[Dict[str, np.ndarray]]) -> DataContainer[Any]:
        """
        Process inference outputs through postprocessing.
        
        Args:
            input_data: Input inference outputs container
            
        Returns:
            Postprocessed results container
        """
        # Extract inference outputs
        raw_outputs = input_data.inference_outputs if input_data.inference_outputs is not None else input_data.data
        
        # Get original shape for coordinate scaling
        original_shape = None
        if input_data.metadata.original_shape is not None:
            # Convert from (H, W, C) to (H, W)
            orig_shape = input_data.metadata.original_shape
            if len(orig_shape) == 3:
                original_shape = (orig_shape[0], orig_shape[1])
            elif len(orig_shape) == 2:
                original_shape = orig_shape
        
        # Run postprocessing
        results = self.postprocessor.postprocess(raw_outputs, original_shape)
        
        # Update metadata
        new_metadata = input_data.metadata.copy()
        new_metadata.update_stage(ProcessingStage.POSTPROCESSING)
        
        # Set appropriate data type based on task
        if self.task_type in ['det', 'detection']:
            new_metadata.data_type = DataType.DETECTION
        elif self.task_type in ['seg', 'segmentation']:
            new_metadata.data_type = DataType.SEGMENTATION
        elif self.task_type in ['kp', 'keypoint', 'pose']:
            new_metadata.data_type = DataType.KEYPOINT
        
        # Create new container
        result_container = DataContainer(
            data=results,
            metadata=new_metadata,
            raw_data=input_data.raw_data,
            preprocessed_data=input_data.preprocessed_data,
            inference_outputs=input_data.inference_outputs,
            postprocessed_results=results
        )
        
        return result_container


class VisualizationAdapter(VisualizationInterface):
    """
    Adapter for visualization components.
    
    This adapter provides visualization capabilities for different result types.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the visualization adapter.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or {}
        self.supported_formats = ['image', 'video', 'json', 'text']
        
        logger.info("Initialized VisualizationAdapter")
    
    def get_stage(self) -> ProcessingStage:
        """Get the processing stage."""
        return ProcessingStage.VISUALIZATION
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        return self.supported_formats
    
    def process(self, input_data: DataContainer[Any]) -> DataContainer[Any]:
        """
        Process results through visualization.
        
        Args:
            input_data: Input results container
            
        Returns:
            Visualization data container
        """
        # Get original image for visualization
        original_image = input_data.raw_data
        if original_image is None:
            raise ValueError("Original image data not available for visualization")
        
        # Get postprocessed results
        results = input_data.postprocessed_results
        if results is None:
            raise ValueError("Postprocessed results not available for visualization")
        
        # Create visualization based on data type
        data_type = input_data.metadata.data_type
        
        if data_type == DataType.DETECTION:
            vis_image = self._visualize_detection(original_image, results)
        elif data_type == DataType.SEGMENTATION:
            vis_image = self._visualize_segmentation(original_image, results)
        elif data_type == DataType.KEYPOINT:
            vis_image = self._visualize_keypoints(original_image, results)
        else:
            # Default: return original image
            vis_image = original_image.copy()
        
        # Update metadata
        new_metadata = input_data.metadata.copy()
        new_metadata.update_stage(ProcessingStage.VISUALIZATION)
        
        # Create new container
        result_container = DataContainer(
            data=vis_image,
            metadata=new_metadata,
            raw_data=input_data.raw_data,
            preprocessed_data=input_data.preprocessed_data,
            inference_outputs=input_data.inference_outputs,
            postprocessed_results=input_data.postprocessed_results,
            visualization_data=vis_image
        )
        
        return result_container
    
    def _visualize_detection(self, image: np.ndarray, results) -> np.ndarray:
        """Visualize detection results."""
        vis_image = image.copy()
        
        # Safely check if results is empty
        try:
            num_detections = len(results)
        except (ValueError, TypeError):
            # Handle cases where len() might fail on arrays
            num_detections = results.boxes.shape[0] if hasattr(results, 'boxes') and results.boxes is not None else 0
        
        if num_detections == 0:
            return vis_image
        
        # Draw bounding boxes
        for i in range(num_detections):
            box = results.boxes[i].astype(int)
            score = results.scores[i]
            class_id = results.class_ids[i]
            
            # Draw rectangle
            cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Draw label
            label = f"Class {class_id}: {score:.2f}"
            cv2.putText(vis_image, label, (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis_image
    
    def _visualize_segmentation(self, image: np.ndarray, results) -> np.ndarray:
        """Visualize segmentation results."""
        vis_image = image.copy()
        
        # Safely check if results is empty
        try:
            num_instances = len(results)
        except (ValueError, TypeError):
            # Handle cases where len() might fail on arrays
            num_instances = results.masks.shape[0] if hasattr(results, 'masks') and results.masks is not None else 0
        
        if num_instances == 0:
            return vis_image
        
        # Overlay masks
        for i in range(num_instances):
            mask = results.masks[i]
            if mask.sum() > 0:
                # Create colored mask
                color = np.random.randint(0, 255, 3)
                colored_mask = np.zeros_like(vis_image)
                colored_mask[mask > 0] = color
                
                # Blend with original image
                vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
        
        # Also draw bounding boxes if available
        if hasattr(results, 'boxes') and getattr(results, 'boxes', None) is not None:
            vis_image = self._visualize_detection(vis_image, results)
        
        return vis_image
    
    def _visualize_keypoints(self, image: np.ndarray, results) -> np.ndarray:
        """Visualize keypoint results."""
        vis_image = image.copy()
        
        # Safely check if results is empty
        try:
            num_persons = len(results)
        except (ValueError, TypeError):
            # Handle cases where len() might fail on arrays
            num_persons = results.keypoints.shape[0] if hasattr(results, 'keypoints') and results.keypoints is not None else 0
        
        if num_persons == 0:
            return vis_image
        
        # Draw keypoints for each person
        for person_idx in range(num_persons):
            keypoints = results.keypoints[person_idx]
            
            # Draw keypoints
            for kp_idx, (x, y, vis) in enumerate(keypoints):
                if vis > 0.5:  # Only draw visible keypoints
                    cv2.circle(vis_image, (int(x), int(y)), 3, (0, 255, 0), -1)
                    cv2.putText(vis_image, str(kp_idx), (int(x) + 5, int(y)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return vis_image


class OutputAdapter(OutputInterface):
    """
    Adapter for output components.
    
    This adapter handles saving results to files, streaming, etc.
    """
    
    def __init__(self, output_path: str, output_format: str = "image"):
        """
        Initialize the output adapter.
        
        Args:
            output_path: Path for output
            output_format: Format for output ('image', 'video', 'json')
        """
        self.output_path = output_path
        self.output_format = output_format
        self.video_writer = None
        
        logger.info(f"Initialized OutputAdapter: {output_format} -> {output_path}")
    
    def get_stage(self) -> ProcessingStage:
        """Get the processing stage."""
        return ProcessingStage.OUTPUT
    
    def process(self, input_data: DataContainer[Any]) -> DataContainer[Any]:
        """
        Process data for output.
        
        Args:
            input_data: Input data container
            
        Returns:
            Output data container
        """
        if self.output_format == "image":
            self._save_image(input_data)
        elif self.output_format == "video":
            self._save_video_frame(input_data)
        elif self.output_format == "json":
            self._save_json(input_data)
        
        # Update metadata
        new_metadata = input_data.metadata.copy()
        new_metadata.update_stage(ProcessingStage.OUTPUT)
        
        # Return the same container with updated metadata
        input_data.metadata = new_metadata
        return input_data
    
    def _save_image(self, input_data: DataContainer[Any]):
        """Save image to file."""
        # Get visualization data or original image - safely handle numpy arrays
        image = None
        if input_data.visualization_data is not None:
            image = input_data.visualization_data
        elif input_data.raw_data is not None:
            image = input_data.raw_data
        
        if image is None:
            raise ValueError("No image data available for saving")
        
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        cv2.imwrite(self.output_path, image_bgr)
        logger.info(f"Saved image to {self.output_path}")
    
    def _save_video_frame(self, input_data: DataContainer[Any]):
        """Save video frame."""
        # Get visualization data or original image - safely handle numpy arrays
        image = None
        if input_data.visualization_data is not None:
            image = input_data.visualization_data
        elif input_data.raw_data is not None:
            image = input_data.raw_data
        
        if image is None:
            return
        
        if self.video_writer is None:
            # Initialize video writer
            height, width = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, 30.0, (width, height))
        
        # Convert RGB to BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        self.video_writer.write(image_bgr)
    
    def _save_json(self, input_data: DataContainer[Any]):
        """Save results as JSON."""
        import json
        
        results = input_data.postprocessed_results
        if results is None:
            return
        
        # Convert results to serializable format
        output_data = {
            'metadata': {
                'source_path': input_data.metadata.source_path,
                'timestamp': input_data.metadata.timestamp,
                'processing_stage': input_data.metadata.processing_stage.value,
                'data_type': input_data.metadata.data_type.value
            },
            'results': self._serialize_results(results)
        }
        
        with open(self.output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved results to {self.output_path}")
    
    def _serialize_results(self, results) -> Dict[str, Any]:
        """Serialize results to JSON-compatible format."""
        if hasattr(results, 'boxes'):
            # Detection or segmentation results
            return {
                'boxes': results.boxes.tolist() if results.boxes is not None else None,
                'scores': results.scores.tolist() if results.scores is not None else None,
                'class_ids': results.class_ids.tolist() if results.class_ids is not None else None,
                'num_detections': len(results)
            }
        elif hasattr(results, 'keypoints'):
            # Keypoint results
            return {
                'keypoints': results.keypoints.tolist() if results.keypoints is not None else None,
                'scores': results.scores.tolist() if results.scores is not None else None,
                'num_persons': len(results)
            }
        else:
            return {'data': str(results)}
    
    def finalize(self) -> None:
        """Finalize output."""
        if self.video_writer is not None:
            self.video_writer.release()
            logger.info(f"Finalized video output: {self.output_path}") 