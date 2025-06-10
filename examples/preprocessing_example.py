"""
Example usage of the image preprocessing module.

This example demonstrates various ways to use the image preprocessing
functionality, including basic usage, custom configurations, and
pipeline composition.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add the parent directory to the path to import hailo_tools
sys.path.append(str(Path(__file__).parent.parent))

from hailo_tools.process import (
    ImagePreprocessor, PreprocessConfig,
    ResizeTransform, NormalizationTransform, 
    PreprocessPipeline
)


def create_sample_image(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a sample image for testing.
    
    Args:
        width (int): Image width
        height (int): Image height
        
    Returns:
        np.ndarray: Sample BGR image
    """
    # Create a colorful test image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some colored rectangles
    cv2.rectangle(image, (50, 50), (200, 200), (255, 0, 0), -1)  # Blue
    cv2.rectangle(image, (250, 50), (400, 200), (0, 255, 0), -1)  # Green
    cv2.rectangle(image, (450, 50), (600, 200), (0, 0, 255), -1)  # Red
    
    # Add some text
    cv2.putText(image, "Sample Image", (200, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add some noise
    noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image


def example_basic_usage():
    """Demonstrate basic preprocessing usage."""
    print("=== Basic Usage Example ===")
    
    # Create sample image
    image = create_sample_image()
    print(f"Original image shape: {image.shape}, dtype: {image.dtype}")
    
    # Create basic configuration
    config = PreprocessConfig(
        target_size=(224, 224),
        normalize=True,
        mean=[0, 0, 0],  # ImageNet means
        std=[1, 1, 1],   # ImageNet stds
        scale=1.0/255.0,             # Scale to [0, 1]
        input_format="BGR",
        output_format="RGB"
    )
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(config)
    
    # Process image
    processed_image = preprocessor(image)
    print(f"Processed image shape: {processed_image.shape}, dtype: {processed_image.dtype}")
    print(f"Processed image range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
    
    return processed_image


def example_custom_pipeline():
    """Demonstrate custom pipeline creation."""
    print("\n=== Custom Pipeline Example ===")
    
    # Create sample image
    image = create_sample_image(800, 600)
    print(f"Original image shape: {image.shape}")
    
    # Create custom transforms
    resize_transform = ResizeTransform(
        target_size=(416, 416),
        interpolation="LINEAR",
        preserve_aspect_ratio=True
    )
    
    normalize_transform = NormalizationTransform(
        mean=127.5,
        std=127.5,
        scale=1.0,
        dtype=np.float32
    )
    
    # Create custom pipeline
    pipeline = PreprocessPipeline(
        transforms=[resize_transform, normalize_transform],
        name="YOLO_Preprocessing",
        enable_timing=True
    )
    
    # Process image
    processed_image = pipeline(image)
    print(f"Processed image shape: {processed_image.shape}, dtype: {processed_image.dtype}")
    print(f"Processed image range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
    
    # Show timing statistics
    pipeline.print_timing_stats()
    
    return processed_image


def example_batch_processing():
    """Demonstrate batch processing."""
    print("\n=== Batch Processing Example ===")
    
    # Create batch of sample images with different sizes
    images = [
        create_sample_image(640, 480),
        create_sample_image(800, 600),
        create_sample_image(1024, 768),
        create_sample_image(512, 384)
    ]
    
    print(f"Batch size: {len(images)}")
    for i, img in enumerate(images):
        print(f"  Image {i}: {img.shape}")
    
    # Create configuration for batch processing
    config = PreprocessConfig(
        target_size=(256, 256),
        normalize=True,
        mean=0.5,
        std=0.5,
        scale=1.0/255.0,
        enable_timing=True
    )
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(config)
    
    # Process batch
    processed_images = preprocessor.process_batch(images)
    
    print(f"Processed batch size: {len(processed_images)}")
    for i, img in enumerate(processed_images):
        print(f"  Processed image {i}: {img.shape}, dtype: {img.dtype}")
    
    # Show timing statistics
    preprocessor.print_timing_stats()
    
    return processed_images


def example_configuration_management():
    """Demonstrate configuration save/load."""
    print("\n=== Configuration Management Example ===")
    
    # Create configuration
    config = PreprocessConfig(
        target_size=(299, 299),
        interpolation="CUBIC",
        preserve_aspect_ratio=False,
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        scale=1.0/255.0,
        target_dtype="float32",
        padding=(10, 10),
        padding_mode="REFLECT",
        input_format="BGR",
        output_format="RGB",
        pipeline_name="InceptionV3_Preprocessing"
    )
    
    # Save configuration
    config_path = Path("preprocessing_config.json")
    config.save(config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Load configuration
    loaded_config = PreprocessConfig.load(config_path)
    print(f"Configuration loaded successfully")
    print(f"Pipeline name: {loaded_config.pipeline_name}")
    print(f"Target size: {loaded_config.target_size}")
    
    # Create preprocessor from loaded config
    preprocessor = ImagePreprocessor(loaded_config)
    
    # Test with sample image
    image = create_sample_image()
    processed_image = preprocessor(image)
    print(f"Processed with loaded config: {processed_image.shape}")
    
    # Clean up
    if config_path.exists():
        config_path.unlink()
    
    return processed_image


def example_dynamic_configuration():
    """Demonstrate dynamic configuration updates."""
    print("\n=== Dynamic Configuration Example ===")
    
    # Create initial preprocessor
    preprocessor = ImagePreprocessor()
    image = create_sample_image()
    
    print("Initial configuration:")
    initial_result = preprocessor(image)
    print(f"  Result shape: {initial_result.shape}, dtype: {initial_result.dtype}")
    
    # Update configuration dynamically
    print("\nUpdating configuration...")
    preprocessor.update_config(
        target_size=(128, 128),
        normalize=True,
        mean=0.0,
        std=255.0,
        target_dtype="float32",
        enable_timing=True
    )
    
    print("Updated configuration:")
    updated_result = preprocessor(image)
    print(f"  Result shape: {updated_result.shape}, dtype: {updated_result.dtype}")
    print(f"  Result range: [{updated_result.min():.3f}, {updated_result.max():.3f}]")
    
    # Show timing stats
    preprocessor.print_timing_stats()
    
    return updated_result


def example_error_handling():
    """Demonstrate error handling."""
    print("\n=== Error Handling Example ===")
    
    try:
        # Try to create invalid configuration
        invalid_config = PreprocessConfig(
            target_size=(0, 224),  # Invalid size
        )
    except Exception as e:
        print(f"Caught expected error: {e}")
    
    try:
        # Try to process invalid input
        preprocessor = ImagePreprocessor()
        result = preprocessor("not an image")  # Invalid input type
    except Exception as e:
        print(f"Caught expected error: {e}")
    
    try:
        # Try invalid interpolation method
        invalid_config = PreprocessConfig(
            target_size=(224, 224),
            interpolation="INVALID_METHOD"
        )
    except Exception as e:
        print(f"Caught expected error: {e}")


def main():
    """Run all examples."""
    print("Image Preprocessing Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_custom_pipeline()
    example_batch_processing()
    example_configuration_management()
    example_dynamic_configuration()
    example_error_handling()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main() 