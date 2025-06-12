"""
Example usage of YOLOv8 postprocessors.

This script demonstrates how to use the detection, segmentation, and keypoint postprocessors
with various configurations and input formats.
"""

import numpy as np
from typing import Dict, Any

from .base import PostprocessConfig
from . import create_postprocessor


def create_mock_detection_output(
    num_detections: int = 10, num_classes: int = 80
) -> Dict[str, np.ndarray]:
    """
    Create mock YOLOv8 detection output for testing.

    Args:
        num_detections: Number of detections to generate
        num_classes: Number of classes in the model

    Returns:
        Dictionary containing mock detection output
    """
    # YOLOv8 detection format: [x_center, y_center, width, height, class_0_conf, class_1_conf, ...]
    num_features = 4 + num_classes

    np.random.seed(42)
    detection_output = np.random.rand(num_detections, num_features)

    # Set some detections to have high confidence
    detection_output[:5, 4:6] = np.random.uniform(0.7, 0.95, (5, 2))  # High confidence
    detection_output[5:, 4:] = np.random.uniform(
        0.1, 0.3, (num_detections - 5, num_classes)
    )  # Low confidence

    # Set reasonable bounding box coordinates (normalized to input size)
    detection_output[:, 0] = np.random.uniform(50, 590, num_detections)  # x_center
    detection_output[:, 1] = np.random.uniform(50, 590, num_detections)  # y_center
    detection_output[:, 2] = np.random.uniform(20, 100, num_detections)  # width
    detection_output[:, 3] = np.random.uniform(20, 100, num_detections)  # height

    return {"output": detection_output}


def create_mock_segmentation_output(
    num_detections: int = 5, num_classes: int = 80
) -> Dict[str, np.ndarray]:
    """
    Create mock YOLOv8 segmentation output for testing.

    Args:
        num_detections: Number of detections to generate
        num_classes: Number of classes in the model

    Returns:
        Dictionary containing mock segmentation output
    """
    # YOLOv8 segmentation format: [x, y, w, h, class_confs..., mask_coeffs...]
    num_mask_coeffs = 32
    num_features = 4 + num_classes + num_mask_coeffs

    np.random.seed(42)
    detection_output = np.random.rand(num_detections, num_features)

    # Set high confidence for some classes
    detection_output[:, 4:6] = np.random.uniform(0.7, 0.9, (num_detections, 2))

    # Set reasonable bounding box coordinates
    detection_output[:, 0] = np.random.uniform(100, 500, num_detections)
    detection_output[:, 1] = np.random.uniform(100, 500, num_detections)
    detection_output[:, 2] = np.random.uniform(50, 150, num_detections)
    detection_output[:, 3] = np.random.uniform(50, 150, num_detections)

    # Create mock mask prototypes
    proto_h, proto_w = 160, 160
    mask_prototypes = np.random.randn(num_mask_coeffs, proto_h, proto_w)

    return {"output0": detection_output, "output1": mask_prototypes}


def create_mock_keypoint_output(
    num_persons: int = 3, num_keypoints: int = 17
) -> Dict[str, np.ndarray]:
    """
    Create mock YOLOv8 keypoint output for testing.

    Args:
        num_persons: Number of person detections to generate
        num_keypoints: Number of keypoints per person

    Returns:
        Dictionary containing mock keypoint output
    """
    # YOLOv8 keypoint format: [x, y, w, h, person_conf, kp1_x, kp1_y, kp1_vis, ...]
    num_features = 5 + num_keypoints * 3

    np.random.seed(42)
    keypoint_output = np.random.rand(num_persons, num_features)

    # Set person confidence scores
    keypoint_output[:, 4] = np.random.uniform(0.7, 0.95, num_persons)

    # Set reasonable person bounding boxes
    keypoint_output[:, 0] = np.random.uniform(200, 400, num_persons)  # x_center
    keypoint_output[:, 1] = np.random.uniform(150, 450, num_persons)  # y_center
    keypoint_output[:, 2] = np.random.uniform(80, 120, num_persons)  # width
    keypoint_output[:, 3] = np.random.uniform(150, 200, num_persons)  # height

    # Set keypoint coordinates and visibility
    for i in range(num_persons):
        kp_start = 5
        for j in range(num_keypoints):
            kp_idx = kp_start + j * 3
            # Generate keypoints within person bounding box
            person_x = keypoint_output[i, 0]
            person_y = keypoint_output[i, 1]
            person_w = keypoint_output[i, 2]
            person_h = keypoint_output[i, 3]

            keypoint_output[i, kp_idx] = person_x + np.random.uniform(
                -person_w / 2, person_w / 2
            )  # x
            keypoint_output[i, kp_idx + 1] = person_y + np.random.uniform(
                -person_h / 2, person_h / 2
            )  # y
            keypoint_output[i, kp_idx + 2] = np.random.uniform(0.3, 0.9)  # visibility

    return {"output": keypoint_output}


def example_detection_postprocessing():
    """Demonstrate detection postprocessing with various configurations."""
    print("=== Detection Postprocessing Example ===")

    # Create configuration for detection
    config = PostprocessConfig(
        num_classes=80,
        det_conf_threshold=0.25,
        nms_iou_threshold=0.5,
        det_max_detections=100,
        input_shape=(640, 640),
        class_names=[f"class_{i}" for i in range(80)],
    )

    # Create postprocessor
    postprocessor = create_postprocessor("detection", config)

    # Create mock detection output
    raw_outputs = create_mock_detection_output()

    # Process detections
    result = postprocessor.postprocess(raw_outputs)

    print(f"Number of detections: {len(result)}")
    if len(result) > 0:
        print(f"Detection boxes shape: {result.boxes.shape}")
        print(f"Confidence scores: {result.scores[:5]}")  # Show first 5
        print(f"Class IDs: {result.class_ids[:5]}")
        print(f"Class names: {[config.class_names[i] for i in result.class_ids[:5]]}")

    # Test with coordinate scaling
    original_shape = (1280, 1280)
    result_scaled = postprocessor.postprocess(raw_outputs, original_shape)

    print(f"\nWith coordinate scaling to {original_shape}:")
    print(f"Number of detections: {len(result_scaled)}")
    if len(result_scaled) > 0:
        print(f"Max coordinate value: {np.max(result_scaled.boxes)}")

    print()


def example_segmentation_postprocessing():
    """Demonstrate segmentation postprocessing with various configurations."""
    print("=== Segmentation Postprocessing Example ===")

    # Create configuration for segmentation
    config = PostprocessConfig(
        num_classes=80,
        seg_conf_threshold=0.25,
        seg_mask_threshold=0.5,
        nms_iou_threshold=0.5,
        seg_max_instances=100,
        input_shape=(640, 640),
        class_names=[f"class_{i}" for i in range(80)],
    )

    # Create postprocessor
    postprocessor = create_postprocessor("segmentation", config)

    # Create mock segmentation output
    raw_outputs = create_mock_segmentation_output()

    # Process segmentations
    result = postprocessor.postprocess(raw_outputs)

    print(f"Number of instances: {len(result)}")
    if len(result) > 0:
        print(f"Mask shape: {result.masks.shape}")
        print(f"Detection boxes shape: {result.boxes.shape}")
        print(f"Confidence scores: {result.scores[:3]}")  # Show first 3
        print(f"Class IDs: {result.class_ids[:3]}")
        print(
            f"Mask coverage (% of pixels): {[np.mean(mask > 0) * 100 for mask in result.masks[:3]]}"
        )

    print()


def example_keypoint_postprocessing():
    """Demonstrate keypoint postprocessing with various configurations."""
    print("=== Keypoint Postprocessing Example ===")

    # Create configuration for keypoint detection
    config = PostprocessConfig(
        num_keypoints=17,  # COCO format
        kp_conf_threshold=0.25,
        kp_visibility_threshold=0.5,
        nms_iou_threshold=0.5,
        kp_max_persons=100,
        input_shape=(640, 640),
        keypoint_names=[
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ],
    )

    # Create postprocessor
    postprocessor = create_postprocessor("keypoint", config)

    # Create mock keypoint output
    raw_outputs = create_mock_keypoint_output()

    # Process keypoints
    result = postprocessor.postprocess(raw_outputs)

    print(f"Number of persons detected: {len(result)}")
    if len(result) > 0:
        print(f"Keypoints shape: {result.keypoints.shape}")
        print(f"Person boxes shape: {result.boxes.shape}")
        print(f"Person confidence scores: {result.scores}")

        # Show keypoint visibility for first person
        if len(result) > 0:
            first_person_kps = result.keypoints[0]
            visible_kps = first_person_kps[
                first_person_kps[:, 2] > config.kp_visibility_threshold
            ]
            print(f"Visible keypoints for first person: {len(visible_kps)}")

            # Show some keypoint names and positions
            for i, (kp_name, kp_data) in enumerate(
                zip(config.keypoint_names, first_person_kps)
            ):
                if kp_data[2] > config.kp_visibility_threshold:  # If visible
                    print(
                        f"  {kp_name}: ({kp_data[0]:.1f}, {kp_data[1]:.1f}) - confidence: {kp_data[2]:.2f}"
                    )
                if i >= 4:  # Show only first 5 visible keypoints
                    break

    print()


def example_custom_configurations():
    """Demonstrate postprocessors with custom configurations."""
    print("=== Custom Configuration Examples ===")

    # High precision detection (strict thresholds)
    print("High precision detection:")
    config_strict = PostprocessConfig(
        num_classes=80,
        det_conf_threshold=0.8,  # High confidence threshold
        nms_iou_threshold=0.3,  # Strict NMS
        det_max_detections=50,  # Fewer detections
        input_shape=(640, 640),
    )

    postprocessor_strict = create_postprocessor("detection", config_strict)
    raw_outputs = create_mock_detection_output()
    result_strict = postprocessor_strict.postprocess(raw_outputs)
    print(f"  Strict detection count: {len(result_strict)}")

    # High recall detection (loose thresholds)
    print("High recall detection:")
    config_loose = PostprocessConfig(
        num_classes=80,
        det_conf_threshold=0.1,  # Low confidence threshold
        nms_iou_threshold=0.7,  # Loose NMS
        det_max_detections=200,  # More detections
        input_shape=(640, 640),
    )

    postprocessor_loose = create_postprocessor("detection", config_loose)
    result_loose = postprocessor_loose.postprocess(raw_outputs)
    print(f"  Loose detection count: {len(result_loose)}")

    # Custom keypoint configuration for specific use case
    print("Custom keypoint configuration:")
    config_kp_custom = PostprocessConfig(
        num_keypoints=17,
        kp_conf_threshold=0.1,  # Low person confidence
        kp_visibility_threshold=0.3,  # Low keypoint visibility threshold
        nms_iou_threshold=0.6,
        kp_max_persons=50,
        input_shape=(640, 640),
    )

    postprocessor_kp_custom = create_postprocessor("keypoint", config_kp_custom)
    kp_outputs = create_mock_keypoint_output()
    result_kp_custom = postprocessor_kp_custom.postprocess(kp_outputs)
    print(f"  Custom keypoint persons: {len(result_kp_custom)}")

    print()


def example_error_handling():
    """Demonstrate error handling in postprocessors."""
    print("=== Error Handling Examples ===")

    try:
        # Invalid configuration
        invalid_config = PostprocessConfig(det_conf_threshold=1.5)  # > 1.0
        print("This should not print - invalid config should raise error")
    except ValueError as e:
        print(f"Caught expected error for invalid config: {e}")

    try:
        # Invalid task type
        config = PostprocessConfig()
        postprocessor = create_postprocessor("invalid_task", config)
        print("This should not print - invalid task should raise error")
    except ValueError as e:
        print(f"Caught expected error for invalid task: {e}")

    try:
        # Empty input
        config = PostprocessConfig()
        postprocessor = create_postprocessor("detection", config)
        result = postprocessor.postprocess({})  # Empty input
        print(f"Empty input handled gracefully: {len(result)} detections")
    except Exception as e:
        print(f"Error with empty input: {e}")

    print()


def main():
    """Run all examples."""
    print("YOLOv8 Postprocessor Examples")
    print("=" * 50)

    example_detection_postprocessing()
    example_segmentation_postprocessing()
    example_keypoint_postprocessing()
    example_custom_configurations()
    example_error_handling()

    print("All examples completed successfully!")


if __name__ == "__main__":
    main()
