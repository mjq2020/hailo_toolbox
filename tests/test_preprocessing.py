"""
Unit tests for the image preprocessing module.

This module contains comprehensive tests for all preprocessing components
including transforms, pipelines, and the main preprocessor class.
"""

import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path
import sys

# Add the parent directory to the path to import hailo_tools
sys.path.append(str(Path(__file__).parent.parent))

from hailo_tools.process import (
    ImagePreprocessor, PreprocessConfig,
    ResizeTransform, NormalizationTransform, DataTypeTransform,
    PaddingTransform, CropTransform, PreprocessPipeline,
    PreprocessError, InvalidConfigError, UnsupportedFormatError,
    InterpolationMethod, PaddingMode
)


class TestPreprocessConfig:
    """Test cases for PreprocessConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = PreprocessConfig()
        assert config.target_size is None
        assert config.interpolation == "LINEAR"
        assert config.normalize is True
        assert config.mean == 0.0
        assert config.std == 1.0
        
    def test_custom_config(self):
        """Test custom configuration creation."""
        config = PreprocessConfig(
            target_size=(224, 224),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            scale=1.0/255.0
        )
        assert config.target_size == (224, 224)
        assert config.mean == [0.485, 0.456, 0.406]
        assert config.std == [0.229, 0.224, 0.225]
        assert config.scale == 1.0/255.0
    
    def test_invalid_target_size(self):
        """Test invalid target size validation."""
        with pytest.raises(InvalidConfigError):
            PreprocessConfig(target_size=(0, 224))
        
        with pytest.raises(InvalidConfigError):
            PreprocessConfig(target_size=(224,))
    
    def test_invalid_interpolation(self):
        """Test invalid interpolation method validation."""
        with pytest.raises(InvalidConfigError):
            PreprocessConfig(interpolation="INVALID")
    
    def test_invalid_std(self):
        """Test invalid standard deviation validation."""
        with pytest.raises(InvalidConfigError):
            PreprocessConfig(std=0.0)
        
        with pytest.raises(InvalidConfigError):
            PreprocessConfig(std=[0.1, 0.0, 0.2])
    
    def test_config_serialization(self):
        """Test configuration save/load functionality."""
        config = PreprocessConfig(
            target_size=(299, 299),
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            pipeline_name="TestConfig"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = Path(f.name)
        
        try:
            # Save and load config
            config.save(config_path)
            loaded_config = PreprocessConfig.load(config_path)
            
            # Verify loaded config
            assert loaded_config.target_size == config.target_size
            assert loaded_config.mean == config.mean
            assert loaded_config.std == config.std
            assert loaded_config.pipeline_name == config.pipeline_name
        finally:
            config_path.unlink()


class TestTransforms:
    """Test cases for individual transform classes."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_gray_image(self):
        """Create a sample grayscale image for testing."""
        return np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    def test_resize_transform(self, sample_image):
        """Test ResizeTransform functionality."""
        transform = ResizeTransform(target_size=(224, 224))
        result = transform(sample_image)
        
        assert result.shape[:2] == (224, 224)
        assert result.shape[2] == sample_image.shape[2]
    
    def test_resize_with_aspect_ratio(self, sample_image):
        """Test ResizeTransform with aspect ratio preservation."""
        transform = ResizeTransform(
            target_size=(224, 224),
            preserve_aspect_ratio=True
        )
        result = transform(sample_image)
        
        assert result.shape[:2] == (224, 224)
    
    def test_resize_invalid_size(self):
        """Test ResizeTransform with invalid target size."""
        with pytest.raises(InvalidConfigError):
            ResizeTransform(target_size=(0, 224))
    
    def test_normalization_transform(self, sample_image):
        """Test NormalizationTransform functionality."""
        transform = NormalizationTransform(
            mean=127.5,
            std=127.5,
            scale=1.0
        )
        result = transform(sample_image)
        
        assert result.dtype == np.float32
        assert np.abs(result.mean()) < 1.0  # Should be approximately normalized
    
    def test_normalization_with_channels(self, sample_image):
        """Test NormalizationTransform with per-channel parameters."""
        transform = NormalizationTransform(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            scale=1.0/255.0
        )
        result = transform(sample_image)
        
        assert result.dtype == np.float32
        assert result.shape == sample_image.shape
    
    def test_data_type_transform(self, sample_image):
        """Test DataTypeTransform functionality."""
        transform = DataTypeTransform(target_dtype=np.float32)
        result = transform(sample_image)
        
        assert result.dtype == np.float32
    
    def test_padding_transform(self, sample_image):
        """Test PaddingTransform functionality."""
        transform = PaddingTransform(padding=10)
        result = transform(sample_image)
        
        expected_shape = (sample_image.shape[0] + 20, 
                         sample_image.shape[1] + 20, 
                         sample_image.shape[2])
        assert result.shape == expected_shape
    
    def test_crop_transform(self, sample_image):
        """Test CropTransform functionality."""
        transform = CropTransform(crop_size=(224, 224), center_crop=True)
        result = transform(sample_image)
        
        assert result.shape[:2] == (224, 224)
        assert result.shape[2] == sample_image.shape[2]
    
    def test_crop_transform_region(self, sample_image):
        """Test CropTransform with specific region."""
        transform = CropTransform(crop_region=(50, 50, 200, 200))
        result = transform(sample_image)
        
        assert result.shape[:2] == (200, 200)
    
    def test_crop_invalid_region(self, sample_image):
        """Test CropTransform with invalid region."""
        transform = CropTransform(crop_region=(0, 0, 1000, 1000))
        
        with pytest.raises(Exception):  # Should raise ImageProcessingError
            transform(sample_image)


class TestPreprocessPipeline:
    """Test cases for PreprocessPipeline class."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_empty_pipeline(self, sample_image):
        """Test pipeline with no transforms."""
        pipeline = PreprocessPipeline()
        result = pipeline(sample_image)
        
        np.testing.assert_array_equal(result, sample_image)
    
    def test_single_transform_pipeline(self, sample_image):
        """Test pipeline with single transform."""
        transform = ResizeTransform(target_size=(224, 224))
        pipeline = PreprocessPipeline([transform])
        result = pipeline(sample_image)
        
        assert result.shape[:2] == (224, 224)
    
    def test_multiple_transforms_pipeline(self, sample_image):
        """Test pipeline with multiple transforms."""
        transforms = [
            ResizeTransform(target_size=(224, 224)),
            NormalizationTransform(mean=127.5, std=127.5)
        ]
        pipeline = PreprocessPipeline(transforms)
        result = pipeline(sample_image)
        
        assert result.shape[:2] == (224, 224)
        assert result.dtype == np.float32
    
    def test_pipeline_timing(self, sample_image):
        """Test pipeline timing functionality."""
        transform = ResizeTransform(target_size=(224, 224))
        pipeline = PreprocessPipeline([transform], enable_timing=True)
        
        # Process image multiple times
        for _ in range(5):
            pipeline(sample_image)
        
        stats = pipeline.get_timing_stats()
        assert "ResizeTransform" in stats
        assert stats["ResizeTransform"]["call_count"] == 5
        assert stats["ResizeTransform"]["total_time"] > 0
    
    def test_pipeline_modification(self, sample_image):
        """Test pipeline modification methods."""
        pipeline = PreprocessPipeline()
        
        # Add transform
        transform1 = ResizeTransform(target_size=(224, 224))
        pipeline.add_transform(transform1)
        assert len(pipeline) == 1
        
        # Insert transform
        transform2 = NormalizationTransform(mean=0, std=1)
        pipeline.insert_transform(0, transform2)
        assert len(pipeline) == 2
        
        # Remove transform
        pipeline.remove_transform(0)
        assert len(pipeline) == 1
    
    def test_batch_processing(self, sample_image):
        """Test batch processing functionality."""
        images = [sample_image, sample_image.copy(), sample_image.copy()]
        
        transform = ResizeTransform(target_size=(224, 224))
        pipeline = PreprocessPipeline([transform])
        
        results = pipeline.process_batch(images)
        
        assert len(results) == len(images)
        for result in results:
            assert result.shape[:2] == (224, 224)


class TestImagePreprocessor:
    """Test cases for ImagePreprocessor class."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_default_preprocessor(self, sample_image):
        """Test preprocessor with default configuration."""
        preprocessor = ImagePreprocessor()
        result = preprocessor(sample_image)
        
        # Default config should normalize the image
        assert result.dtype == np.float32
    
    def test_custom_config_preprocessor(self, sample_image):
        """Test preprocessor with custom configuration."""
        config = PreprocessConfig(
            target_size=(224, 224),
            normalize=True,
            mean=0.5,
            std=0.5,
            scale=1.0/255.0
        )
        
        preprocessor = ImagePreprocessor(config)
        result = preprocessor(sample_image)
        
        assert result.shape[:2] == (224, 224)
        assert result.dtype == np.float32
    
    def test_color_format_conversion(self, sample_image):
        """Test color format conversion."""
        config = PreprocessConfig(
            input_format="BGR",
            output_format="RGB"
        )
        
        preprocessor = ImagePreprocessor(config)
        result = preprocessor(sample_image)
        
        # Should convert BGR to RGB
        assert result.shape == sample_image.shape
    
    def test_grayscale_conversion(self, sample_image):
        """Test grayscale conversion."""
        config = PreprocessConfig(
            input_format="BGR",
            output_format="GRAY"
        )
        
        preprocessor = ImagePreprocessor(config)
        result = preprocessor(sample_image)
        
        # Should convert to grayscale
        assert len(result.shape) == 2
    
    def test_dynamic_config_update(self, sample_image):
        """Test dynamic configuration updates."""
        preprocessor = ImagePreprocessor()
        
        # Initial processing
        result1 = preprocessor(sample_image)
        
        # Update configuration
        preprocessor.update_config(
            target_size=(128, 128),
            normalize=False
        )
        
        # Process again
        result2 = preprocessor(sample_image)
        
        assert result2.shape[:2] == (128, 128)
        assert result1.shape != result2.shape
    
    def test_batch_processing(self, sample_image):
        """Test batch processing."""
        images = [sample_image, sample_image.copy()]
        
        config = PreprocessConfig(target_size=(224, 224))
        preprocessor = ImagePreprocessor(config)
        
        results = preprocessor.process_batch(images)
        
        assert len(results) == len(images)
        for result in results:
            assert result.shape[:2] == (224, 224)
    
    def test_invalid_input(self):
        """Test preprocessor with invalid input."""
        preprocessor = ImagePreprocessor()
        
        with pytest.raises(PreprocessError):
            preprocessor("not an image")
        
        with pytest.raises(PreprocessError):
            preprocessor(np.array([]))
    
    def test_config_file_loading(self, sample_image):
        """Test loading preprocessor from config file."""
        config = PreprocessConfig(
            target_size=(256, 256),
            pipeline_name="TestPreprocessor"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = Path(f.name)
        
        try:
            # Save config and create preprocessor from file
            config.save(config_path)
            preprocessor = ImagePreprocessor.from_config_file(config_path)
            
            result = preprocessor(sample_image)
            assert result.shape[:2] == (256, 256)
        finally:
            config_path.unlink()


class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_unsupported_format_error(self):
        """Test UnsupportedFormatError."""
        error = UnsupportedFormatError(
            "Test error",
            format_type="INVALID",
            supported_formats=["BGR", "RGB"]
        )
        
        assert "Test error" in str(error)
        assert error.format_type == "INVALID"
        assert error.supported_formats == ["BGR", "RGB"]
    
    def test_invalid_config_error(self):
        """Test InvalidConfigError."""
        error = InvalidConfigError(
            "Invalid config",
            config_field="test_field",
            provided_value="invalid_value"
        )
        
        assert "Invalid config" in str(error)
        assert error.config_field == "test_field"
        assert error.provided_value == "invalid_value"
    
    def test_preprocess_error(self):
        """Test PreprocessError."""
        error = PreprocessError(
            "Processing failed",
            details={"operation": "test", "value": 123}
        )
        
        assert "Processing failed" in str(error)
        assert error.details["operation"] == "test"
        assert error.details["value"] == 123


def test_integration():
    """Integration test for the entire preprocessing system."""
    # Create a complex preprocessing pipeline
    config = PreprocessConfig(
        target_size=(224, 224),
        interpolation="CUBIC",
        preserve_aspect_ratio=True,
        normalize=True,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        scale=1.0/255.0,
        input_format="BGR",
        output_format="RGB",
        enable_timing=True
    )
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(config)
    
    # Create test image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Process image
    result = preprocessor(image)
    
    # Verify result
    assert result.shape[:2] == (224, 224)
    assert result.dtype == np.float32
    assert result.shape[2] == 3  # RGB output
    
    # Check that timing was recorded
    stats = preprocessor.get_timing_stats()
    assert len(stats) > 0


if __name__ == "__main__":
    pytest.main([__file__]) 