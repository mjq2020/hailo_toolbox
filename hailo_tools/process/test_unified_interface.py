"""
Unit tests for the unified data interface system.

This module contains comprehensive tests for all components of the unified
data interface, including data containers, pipeline components, and adapters.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from .interface import (
    DataType, ProcessingStage, Metadata, DataContainer,
    Pipeline, PipelineConfig, create_image_container,
    create_batch_container, create_tensor_container
)
from .adapters import (
    ImageSourceAdapter, PreprocessorAdapter, InferenceAdapter,
    PostprocessorAdapter, VisualizationAdapter, OutputAdapter
)
from .base import PostprocessConfig
from .preprocessor import PreprocessConfig


class TestDataTypes(unittest.TestCase):
    """Test data type enumerations and basic structures."""
    
    def test_data_type_enum(self):
        """Test DataType enumeration values."""
        self.assertEqual(DataType.IMAGE.value, "image")
        self.assertEqual(DataType.TENSOR.value, "tensor")
        self.assertEqual(DataType.DETECTION_RESULT.value, "detection_result")
        self.assertEqual(DataType.SEGMENTATION_RESULT.value, "segmentation_result")
        self.assertEqual(DataType.KEYPOINT_RESULT.value, "keypoint_result")
    
    def test_processing_stage_enum(self):
        """Test ProcessingStage enumeration values."""
        self.assertEqual(ProcessingStage.SOURCE.value, "source")
        self.assertEqual(ProcessingStage.PREPROCESSING.value, "preprocessing")
        self.assertEqual(ProcessingStage.INFERENCE.value, "inference")
        self.assertEqual(ProcessingStage.POSTPROCESSING.value, "postprocessing")
        self.assertEqual(ProcessingStage.VISUALIZATION.value, "visualization")
        self.assertEqual(ProcessingStage.OUTPUT.value, "output")


class TestMetadata(unittest.TestCase):
    """Test metadata functionality."""
    
    def test_metadata_creation(self):
        """Test metadata object creation."""
        metadata = Metadata(
            data_type=DataType.IMAGE,
            source_info={'path': 'test.jpg', 'type': 'file'},
            processing_stage=ProcessingStage.SOURCE
        )
        
        self.assertEqual(metadata.data_type, DataType.IMAGE)
        self.assertEqual(metadata.source_info['path'], 'test.jpg')
        self.assertEqual(metadata.processing_stage, ProcessingStage.SOURCE)
        self.assertIsInstance(metadata.timestamp, float)
        self.assertEqual(len(metadata.processing_history), 0)
        self.assertEqual(len(metadata.processing_times), 0)
    
    def test_metadata_add_processing_step(self):
        """Test adding processing steps to metadata."""
        metadata = Metadata(
            data_type=DataType.IMAGE,
            processing_stage=ProcessingStage.SOURCE
        )
        
        metadata.add_processing_step(ProcessingStage.PREPROCESSING, 0.1)
        
        self.assertEqual(len(metadata.processing_history), 1)
        self.assertEqual(metadata.processing_history[0], ProcessingStage.PREPROCESSING)
        self.assertEqual(metadata.processing_times[ProcessingStage.PREPROCESSING.value], 0.1)
        self.assertEqual(metadata.processing_stage, ProcessingStage.PREPROCESSING)
    
    def test_metadata_update_performance(self):
        """Test updating performance metrics."""
        metadata = Metadata(
            data_type=DataType.IMAGE,
            processing_stage=ProcessingStage.SOURCE
        )
        
        metadata.update_performance({'fps': 30.0, 'latency': 0.033})
        
        self.assertEqual(metadata.performance_metrics['fps'], 30.0)
        self.assertEqual(metadata.performance_metrics['latency'], 0.033)


class TestDataContainer(unittest.TestCase):
    """Test data container functionality."""
    
    def test_data_container_creation(self):
        """Test data container creation."""
        test_data = np.random.rand(640, 640, 3)
        metadata = Metadata(
            data_type=DataType.IMAGE,
            processing_stage=ProcessingStage.SOURCE
        )
        
        container = DataContainer(test_data, metadata)
        
        self.assertTrue(np.array_equal(container.data, test_data))
        self.assertEqual(container.metadata.data_type, DataType.IMAGE)
        self.assertEqual(container.metadata.processing_stage, ProcessingStage.SOURCE)
    
    def test_data_container_update_stage(self):
        """Test updating processing stage."""
        test_data = np.random.rand(640, 640, 3)
        metadata = Metadata(
            data_type=DataType.IMAGE,
            processing_stage=ProcessingStage.SOURCE
        )
        container = DataContainer(test_data, metadata)
        
        container.update_stage(ProcessingStage.PREPROCESSING, 0.05)
        
        self.assertEqual(container.metadata.processing_stage, ProcessingStage.PREPROCESSING)
        self.assertEqual(len(container.metadata.processing_history), 1)
        self.assertEqual(container.metadata.processing_times['preprocessing'], 0.05)
    
    def test_data_container_clone(self):
        """Test cloning data container."""
        test_data = np.random.rand(640, 640, 3)
        metadata = Metadata(
            data_type=DataType.IMAGE,
            processing_stage=ProcessingStage.SOURCE
        )
        container = DataContainer(test_data, metadata)
        
        cloned = container.clone()
        
        self.assertTrue(np.array_equal(cloned.data, container.data))
        self.assertEqual(cloned.metadata.data_type, container.metadata.data_type)
        self.assertIsNot(cloned.metadata, container.metadata)  # Different objects


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for creating data containers."""
    
    def test_create_image_container(self):
        """Test creating image container."""
        image = np.random.rand(640, 640, 3).astype(np.uint8)
        source_info = {'path': 'test.jpg', 'type': 'file'}
        
        container = create_image_container(image, source_info)
        
        self.assertTrue(np.array_equal(container.data, image))
        self.assertEqual(container.metadata.data_type, DataType.IMAGE)
        self.assertEqual(container.metadata.processing_stage, ProcessingStage.SOURCE)
        self.assertEqual(container.metadata.source_info, source_info)
    
    def test_create_batch_container(self):
        """Test creating batch container."""
        batch = np.random.rand(4, 3, 640, 640)
        
        container = create_batch_container(batch)
        
        self.assertTrue(np.array_equal(container.data, batch))
        self.assertEqual(container.metadata.data_type, DataType.BATCH)
        self.assertEqual(container.metadata.processing_stage, ProcessingStage.PREPROCESSING)
    
    def test_create_tensor_container(self):
        """Test creating tensor container."""
        tensor = np.random.rand(1, 3, 640, 640)
        
        container = create_tensor_container(tensor)
        
        self.assertTrue(np.array_equal(container.data, tensor))
        self.assertEqual(container.metadata.data_type, DataType.TENSOR)
        self.assertEqual(container.metadata.processing_stage, ProcessingStage.PREPROCESSING)


class TestPipelineConfig(unittest.TestCase):
    """Test pipeline configuration."""
    
    def test_pipeline_config_creation(self):
        """Test pipeline configuration creation."""
        config = PipelineConfig(
            batch_size=4,
            enable_profiling=True,
            continue_on_error=False
        )
        
        self.assertEqual(config.batch_size, 4)
        self.assertTrue(config.enable_profiling)
        self.assertFalse(config.continue_on_error)
    
    def test_pipeline_config_defaults(self):
        """Test pipeline configuration defaults."""
        config = PipelineConfig()
        
        self.assertEqual(config.batch_size, 1)
        self.assertFalse(config.enable_profiling)
        self.assertTrue(config.continue_on_error)


class TestImageSourceAdapter(unittest.TestCase):
    """Test image source adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary image file
        self.temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        self.temp_image_path = self.temp_image.name
        self.temp_image.close()
        
        # Create a mock image and save it
        import cv2
        mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(self.temp_image_path, mock_image)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_image_path):
            os.unlink(self.temp_image_path)
    
    def test_image_source_adapter_creation(self):
        """Test image source adapter creation."""
        adapter = ImageSourceAdapter(self.temp_image_path, "file")
        
        self.assertEqual(adapter.source_path, self.temp_image_path)
        self.assertEqual(adapter.source_type, "file")
        self.assertTrue(adapter.is_available())
    
    def test_image_source_adapter_properties(self):
        """Test image source adapter properties."""
        adapter = ImageSourceAdapter(self.temp_image_path, "file")
        properties = adapter.get_properties()
        
        self.assertIn('width', properties)
        self.assertIn('height', properties)
        self.assertIn('channels', properties)
        self.assertEqual(properties['width'], 640)
        self.assertEqual(properties['height'], 480)
        self.assertEqual(properties['channels'], 3)
    
    def test_image_source_adapter_process(self):
        """Test image source adapter processing."""
        adapter = ImageSourceAdapter(self.temp_image_path, "file")
        result = adapter.process(None)
        
        self.assertIsInstance(result, DataContainer)
        self.assertEqual(result.metadata.data_type, DataType.IMAGE)
        self.assertEqual(result.metadata.processing_stage, ProcessingStage.SOURCE)
        self.assertEqual(result.data.shape, (480, 640, 3))


class TestPreprocessorAdapter(unittest.TestCase):
    """Test preprocessor adapter."""
    
    def test_preprocessor_adapter_creation(self):
        """Test preprocessor adapter creation."""
        config = PreprocessConfig(target_width=640, target_height=640)
        adapter = PreprocessorAdapter(config)
        
        self.assertEqual(adapter.config.target_width, 640)
        self.assertEqual(adapter.config.target_height, 640)
    
    def test_preprocessor_adapter_input_spec(self):
        """Test preprocessor adapter input specification."""
        config = PreprocessConfig(target_width=640, target_height=640)
        adapter = PreprocessorAdapter(config)
        
        input_spec = adapter.get_input_spec()
        self.assertEqual(input_spec['data_type'], DataType.IMAGE)
        self.assertEqual(input_spec['required_stage'], ProcessingStage.SOURCE)
    
    def test_preprocessor_adapter_output_spec(self):
        """Test preprocessor adapter output specification."""
        config = PreprocessConfig(target_width=640, target_height=640)
        adapter = PreprocessorAdapter(config)
        
        output_spec = adapter.get_output_spec()
        self.assertEqual(output_spec['data_type'], DataType.TENSOR)
        self.assertEqual(output_spec['output_stage'], ProcessingStage.PREPROCESSING)


class TestInferenceAdapter(unittest.TestCase):
    """Test inference adapter."""
    
    def test_inference_adapter_creation(self):
        """Test inference adapter creation."""
        mock_engine = Mock()
        model_info = {'name': 'YOLOv8', 'version': '1.0', 'task': 'detection'}
        
        adapter = InferenceAdapter(mock_engine, model_info)
        
        self.assertEqual(adapter.engine, mock_engine)
        self.assertEqual(adapter.model_info, model_info)
    
    def test_inference_adapter_warm_up(self):
        """Test inference adapter warm up."""
        mock_engine = Mock()
        mock_engine.warm_up = Mock()
        model_info = {'name': 'YOLOv8', 'version': '1.0', 'task': 'detection'}
        
        adapter = InferenceAdapter(mock_engine, model_info)
        adapter.warm_up((1, 3, 640, 640))
        
        mock_engine.warm_up.assert_called_once_with((1, 3, 640, 640))
    
    def test_inference_adapter_process(self):
        """Test inference adapter processing."""
        mock_engine = Mock()
        mock_engine.infer = Mock(return_value={'output': np.random.rand(1, 10, 85)})
        model_info = {'name': 'YOLOv8', 'version': '1.0', 'task': 'detection'}
        
        adapter = InferenceAdapter(mock_engine, model_info)
        
        # Create input tensor container
        input_tensor = np.random.rand(1, 3, 640, 640)
        input_metadata = Metadata(
            data_type=DataType.TENSOR,
            processing_stage=ProcessingStage.PREPROCESSING
        )
        input_container = DataContainer(input_tensor, input_metadata)
        
        result = adapter.process(input_container)
        
        self.assertIsInstance(result, DataContainer)
        self.assertEqual(result.metadata.data_type, DataType.TENSOR)
        self.assertEqual(result.metadata.processing_stage, ProcessingStage.INFERENCE)
        mock_engine.infer.assert_called_once_with(input_tensor)


class TestPostprocessorAdapter(unittest.TestCase):
    """Test postprocessor adapter."""
    
    def test_postprocessor_adapter_creation(self):
        """Test postprocessor adapter creation."""
        config = PostprocessConfig(num_classes=80, det_conf_threshold=0.25)
        adapter = PostprocessorAdapter("detection", config)
        
        self.assertEqual(adapter.task_type, "detection")
        self.assertEqual(adapter.config.num_classes, 80)
        self.assertEqual(adapter.config.det_conf_threshold, 0.25)
    
    def test_postprocessor_adapter_input_spec(self):
        """Test postprocessor adapter input specification."""
        config = PostprocessConfig(num_classes=80)
        adapter = PostprocessorAdapter("detection", config)
        
        input_spec = adapter.get_input_spec()
        self.assertEqual(input_spec['data_type'], DataType.TENSOR)
        self.assertEqual(input_spec['required_stage'], ProcessingStage.INFERENCE)
    
    def test_postprocessor_adapter_output_spec(self):
        """Test postprocessor adapter output specification."""
        config = PostprocessConfig(num_classes=80)
        adapter = PostprocessorAdapter("detection", config)
        
        output_spec = adapter.get_output_spec()
        self.assertEqual(output_spec['data_type'], DataType.DETECTION_RESULT)
        self.assertEqual(output_spec['output_stage'], ProcessingStage.POSTPROCESSING)


class TestVisualizationAdapter(unittest.TestCase):
    """Test visualization adapter."""
    
    def test_visualization_adapter_creation(self):
        """Test visualization adapter creation."""
        adapter = VisualizationAdapter()
        
        self.assertIsNotNone(adapter)
    
    def test_visualization_adapter_input_spec(self):
        """Test visualization adapter input specification."""
        adapter = VisualizationAdapter()
        
        input_spec = adapter.get_input_spec()
        self.assertIn(input_spec['data_type'], [
            DataType.DETECTION_RESULT,
            DataType.SEGMENTATION_RESULT,
            DataType.KEYPOINT_RESULT
        ])
        self.assertEqual(input_spec['required_stage'], ProcessingStage.POSTPROCESSING)
    
    def test_visualization_adapter_output_spec(self):
        """Test visualization adapter output specification."""
        adapter = VisualizationAdapter()
        
        output_spec = adapter.get_output_spec()
        self.assertEqual(output_spec['data_type'], DataType.IMAGE)
        self.assertEqual(output_spec['output_stage'], ProcessingStage.VISUALIZATION)


class TestOutputAdapter(unittest.TestCase):
    """Test output adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_output = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        self.temp_output_path = self.temp_output.name
        self.temp_output.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_output_path):
            os.unlink(self.temp_output_path)
    
    def test_output_adapter_creation(self):
        """Test output adapter creation."""
        adapter = OutputAdapter(self.temp_output_path, "image")
        
        self.assertEqual(adapter.output_path, self.temp_output_path)
        self.assertEqual(adapter.output_format, "image")
    
    def test_output_adapter_input_spec(self):
        """Test output adapter input specification."""
        adapter = OutputAdapter(self.temp_output_path, "image")
        
        input_spec = adapter.get_input_spec()
        self.assertEqual(input_spec['data_type'], DataType.IMAGE)
        self.assertEqual(input_spec['required_stage'], ProcessingStage.VISUALIZATION)
    
    def test_output_adapter_output_spec(self):
        """Test output adapter output specification."""
        adapter = OutputAdapter(self.temp_output_path, "image")
        
        output_spec = adapter.get_output_spec()
        self.assertEqual(output_spec['data_type'], DataType.IMAGE)
        self.assertEqual(output_spec['output_stage'], ProcessingStage.OUTPUT)


class TestPipeline(unittest.TestCase):
    """Test pipeline functionality."""
    
    def test_pipeline_creation(self):
        """Test pipeline creation."""
        config = PipelineConfig()
        pipeline = Pipeline(config)
        
        self.assertEqual(pipeline.config, config)
        self.assertEqual(len(pipeline.components), 0)
        self.assertFalse(pipeline.is_initialized)
    
    def test_pipeline_add_component(self):
        """Test adding components to pipeline."""
        config = PipelineConfig()
        pipeline = Pipeline(config)
        
        mock_component = Mock()
        mock_component.get_input_spec.return_value = {
            'data_type': DataType.IMAGE,
            'required_stage': ProcessingStage.SOURCE
        }
        mock_component.get_output_spec.return_value = {
            'data_type': DataType.TENSOR,
            'output_stage': ProcessingStage.PREPROCESSING
        }
        
        result = pipeline.add_component(mock_component)
        
        self.assertEqual(result, pipeline)  # Should return self for chaining
        self.assertEqual(len(pipeline.components), 1)
    
    def test_pipeline_initialize(self):
        """Test pipeline initialization."""
        config = PipelineConfig()
        pipeline = Pipeline(config)
        
        # Add mock components
        mock_component1 = Mock()
        mock_component1.get_input_spec.return_value = {
            'data_type': DataType.IMAGE,
            'required_stage': ProcessingStage.SOURCE
        }
        mock_component1.get_output_spec.return_value = {
            'data_type': DataType.TENSOR,
            'output_stage': ProcessingStage.PREPROCESSING
        }
        mock_component1.initialize = Mock()
        
        mock_component2 = Mock()
        mock_component2.get_input_spec.return_value = {
            'data_type': DataType.TENSOR,
            'required_stage': ProcessingStage.PREPROCESSING
        }
        mock_component2.get_output_spec.return_value = {
            'data_type': DataType.TENSOR,
            'output_stage': ProcessingStage.INFERENCE
        }
        mock_component2.initialize = Mock()
        
        pipeline.add_component(mock_component1)
        pipeline.add_component(mock_component2)
        
        pipeline.initialize()
        
        self.assertTrue(pipeline.is_initialized)
        mock_component1.initialize.assert_called_once()
        mock_component2.initialize.assert_called_once()
    
    def test_pipeline_get_statistics(self):
        """Test getting pipeline statistics."""
        config = PipelineConfig()
        pipeline = Pipeline(config)
        
        stats = pipeline.get_statistics()
        
        self.assertIn('total_processed', stats)
        self.assertIn('total_errors', stats)
        self.assertIn('average_processing_time', stats)
        self.assertEqual(stats['total_processed'], 0)
        self.assertEqual(stats['total_errors'], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the unified interface system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary image file
        self.temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        self.temp_image_path = self.temp_image.name
        self.temp_image.close()
        
        # Create a mock image and save it
        import cv2
        mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(self.temp_image_path, mock_image)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_image_path):
            os.unlink(self.temp_image_path)
    
    @patch('hailo_tools.process.adapters.ImagePreprocessor')
    def test_simple_pipeline_integration(self, mock_preprocessor_class):
        """Test a simple pipeline integration."""
        # Mock the preprocessor
        mock_preprocessor = Mock()
        mock_preprocessor.preprocess.return_value = np.random.rand(1, 3, 640, 640)
        mock_preprocessor_class.return_value = mock_preprocessor
        
        # Mock inference engine
        mock_engine = Mock()
        mock_engine.infer.return_value = {'output': np.random.rand(1, 10, 85)}
        
        # Create pipeline configuration
        config = PipelineConfig()
        
        # Create pipeline components
        source = ImageSourceAdapter(self.temp_image_path, "file")
        preprocessor = PreprocessorAdapter(PreprocessConfig(
            target_width=640,
            target_height=640
        ))
        inference = InferenceAdapter(mock_engine, {
            'name': 'TestModel',
            'version': '1.0',
            'task': 'detection'
        })
        
        # Build pipeline
        pipeline = Pipeline(config)
        pipeline.add_component(source)
        pipeline.add_component(preprocessor)
        pipeline.add_component(inference)
        
        # Initialize pipeline
        pipeline.initialize()
        
        # Verify pipeline is properly set up
        self.assertTrue(pipeline.is_initialized)
        self.assertEqual(len(pipeline.components), 3)
    
    def test_error_handling_integration(self):
        """Test error handling in pipeline integration."""
        config = PipelineConfig(continue_on_error=True)
        
        # Create a source that will fail
        source = ImageSourceAdapter("/non/existent/path.jpg", "file")
        
        pipeline = Pipeline(config)
        pipeline.add_component(source)
        
        # This should not raise an exception due to continue_on_error=True
        try:
            pipeline.initialize()
            # The pipeline should handle the error gracefully
            self.assertTrue(True)  # If we get here, error handling worked
        except Exception as e:
            self.fail(f"Pipeline should have handled error gracefully: {e}")


def run_tests():
    """
    Run all tests for the unified interface system.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataTypes,
        TestMetadata,
        TestDataContainer,
        TestUtilityFunctions,
        TestPipelineConfig,
        TestImageSourceAdapter,
        TestPreprocessorAdapter,
        TestInferenceAdapter,
        TestPostprocessorAdapter,
        TestVisualizationAdapter,
        TestOutputAdapter,
        TestPipeline,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 