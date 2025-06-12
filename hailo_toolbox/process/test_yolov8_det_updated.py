"""
Test script for updated YOLOv8 Detection Postprocessor

This script tests both NMS and non-NMS postprocessing modes with different output formats.
"""

import numpy as np
import unittest
from typing import Dict

from hailo_toolbox.process import YOLOv8DetPostprocessor, PostprocessConfig


class TestUpdatedYOLOv8DetPostprocessor(unittest.TestCase):
    """Test cases for updated YOLOv8 detection postprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_with_nms = PostprocessConfig(
            num_classes=80,
            det_conf_threshold=0.25,
            nms_iou_threshold=0.5,
            det_max_detections=100,
            input_shape=(640, 640),
            nms=True,  # Enable NMS
            det_class_agnostic=False,
        )

        self.config_without_nms = PostprocessConfig(
            num_classes=80,
            det_conf_threshold=0.25,
            det_max_detections=100,
            input_shape=(640, 640),
            nms=False,  # Disable NMS
        )

    def test_non_nms_format_processing(self):
        """Test processing of non-NMS format: [batch, number, 6] where 6 = [x1, y1, x2, y2, conf, label]"""
        postprocessor = YOLOv8DetPostprocessor(self.config_without_nms)

        # Create mock detection output in non-NMS format
        batch_size = 1
        num_detections = 10
        num_features = 6  # [x1, y1, x2, y2, conf, label]

        np.random.seed(42)
        detection_output = np.random.rand(batch_size, num_detections, num_features)

        # Set realistic bounding boxes (x1, y1, x2, y2)
        detection_output[0, :, 0] = np.random.uniform(0, 300, num_detections)  # x1
        detection_output[0, :, 1] = np.random.uniform(0, 300, num_detections)  # y1
        detection_output[0, :, 2] = detection_output[0, :, 0] + np.random.uniform(
            50, 200, num_detections
        )  # x2
        detection_output[0, :, 3] = detection_output[0, :, 1] + np.random.uniform(
            50, 200, num_detections
        )  # y2

        # Set high confidence scores for some detections
        detection_output[0, :5, 4] = np.random.uniform(0.7, 0.9, 5)  # High confidence
        detection_output[0, 5:, 4] = np.random.uniform(0.1, 0.3, 5)  # Low confidence

        # Set class labels
        detection_output[0, :, 5] = np.random.randint(0, 80, num_detections)

        raw_outputs = {"output": detection_output}

        result = postprocessor.postprocess(raw_outputs)

        # Verify results
        self.assertIsNotNone(result)
        self.assertTrue(len(result) > 0)  # Should have some detections above threshold
        self.assertEqual(result.boxes.shape[1], 4)  # x1, y1, x2, y2
        self.assertEqual(len(result.scores), len(result.boxes))
        self.assertEqual(len(result.class_ids), len(result.boxes))

        # Verify box format (x2 > x1, y2 > y1)
        self.assertTrue(np.all(result.boxes[:, 2] > result.boxes[:, 0]))
        self.assertTrue(np.all(result.boxes[:, 3] > result.boxes[:, 1]))

        print(f"Non-NMS format: Processed {len(result)} detections")

    def test_nms_format_processing_single_output(self):
        """Test NMS processing with single output (traditional YOLOv8 format)"""
        postprocessor = YOLOv8DetPostprocessor(self.config_with_nms)

        # Create mock detection output in traditional YOLOv8 format
        batch_size = 1
        num_detections = 15
        num_features = 4 + self.config_with_nms.num_classes  # x, y, w, h + classes

        np.random.seed(42)
        detection_output = np.random.rand(batch_size, num_detections, num_features)

        # Set realistic center coordinates and dimensions
        detection_output[0, :, 0] = np.random.uniform(
            100, 500, num_detections
        )  # center_x
        detection_output[0, :, 1] = np.random.uniform(
            100, 500, num_detections
        )  # center_y
        detection_output[0, :, 2] = np.random.uniform(50, 200, num_detections)  # width
        detection_output[0, :, 3] = np.random.uniform(50, 200, num_detections)  # height

        # Set high class scores for some detections
        detection_output[0, :8, 4:6] = 0.8  # High scores for first 8 detections
        detection_output[0, 8:, 4:] = 0.1  # Low scores for rest

        raw_outputs = {"output": detection_output}

        result = postprocessor.postprocess(raw_outputs)

        # Verify results
        self.assertIsNotNone(result)
        self.assertTrue(len(result) > 0)
        self.assertEqual(result.boxes.shape[1], 4)

        print(f"NMS format (single output): Processed {len(result)} detections")

    def test_nms_format_processing_multiple_outputs(self):
        """Test NMS processing with multiple outputs that need concatenation"""
        postprocessor = YOLOv8DetPostprocessor(self.config_with_nms)

        # Create multiple mock outputs from different detection heads
        batch_size = 1
        num_features = 4 + self.config_with_nms.num_classes

        np.random.seed(42)

        # Create 3 different detection heads with different numbers of detections
        output1 = np.random.rand(batch_size, 5, num_features)  # 5 detections
        output2 = np.random.rand(batch_size, 8, num_features)  # 8 detections
        output3 = np.random.rand(batch_size, 7, num_features)  # 7 detections

        # Set realistic values for each output
        for i, output in enumerate([output1, output2, output3]):
            num_dets = output.shape[1]
            # Center coordinates
            output[0, :, 0] = np.random.uniform(100, 500, num_dets)
            output[0, :, 1] = np.random.uniform(100, 500, num_dets)
            # Dimensions
            output[0, :, 2] = np.random.uniform(50, 200, num_dets)
            output[0, :, 3] = np.random.uniform(50, 200, num_dets)
            # High confidence for some detections
            high_conf_count = num_dets // 2
            output[0, :high_conf_count, 4:6] = 0.8
            output[0, high_conf_count:, 4:] = 0.1

        raw_outputs = {
            "output_head_1": output1,
            "output_head_2": output2,
            "output_head_3": output3,
        }

        result = postprocessor.postprocess(raw_outputs)

        # Verify results
        self.assertIsNotNone(result)
        self.assertTrue(len(result) > 0)
        self.assertEqual(result.boxes.shape[1], 4)

        print(
            f"NMS format (multiple outputs): Processed {len(result)} detections from {len(raw_outputs)} heads"
        )

    def test_coordinate_scaling_non_nms(self):
        """Test coordinate scaling for non-NMS format"""
        postprocessor = YOLOv8DetPostprocessor(self.config_without_nms)

        # Create detection output with known coordinates
        detection_output = np.zeros((1, 3, 6))

        # Set known bounding boxes in input resolution (640x640)
        detection_output[0, 0] = [100, 100, 200, 200, 0.9, 0]  # Box 1
        detection_output[0, 1] = [300, 300, 400, 400, 0.8, 1]  # Box 2
        detection_output[0, 2] = [50, 50, 150, 150, 0.7, 2]  # Box 3

        raw_outputs = {"output": detection_output}
        original_shape = (1280, 1280)  # 2x scaling

        result = postprocessor.postprocess(raw_outputs, original_shape)

        # Verify scaling (should be 2x)
        self.assertEqual(len(result), 3)
        expected_boxes = np.array(
            [
                [200, 200, 400, 400],  # Box 1 scaled
                [600, 600, 800, 800],  # Box 2 scaled
                [100, 100, 300, 300],  # Box 3 scaled
            ]
        )

        np.testing.assert_array_almost_equal(result.boxes, expected_boxes, decimal=1)
        print("Coordinate scaling test passed for non-NMS format")

    def test_confidence_filtering(self):
        """Test confidence threshold filtering"""
        config = PostprocessConfig(
            num_classes=80,
            det_conf_threshold=0.5,  # Higher threshold
            det_max_detections=100,
            nms=False,
        )
        postprocessor = YOLOv8DetPostprocessor(config)

        # Create detections with varying confidence
        detection_output = np.zeros((1, 5, 6))
        detection_output[0, :, :4] = [
            [100, 100, 200, 200],
            [200, 200, 300, 300],
            [300, 300, 400, 400],
            [400, 400, 500, 500],
            [500, 500, 600, 600],
        ]
        detection_output[0, :, 4] = [0.8, 0.6, 0.4, 0.3, 0.7]  # Confidences
        detection_output[0, :, 5] = [0, 1, 2, 3, 4]  # Classes

        raw_outputs = {"output": detection_output}
        result = postprocessor.postprocess(raw_outputs)

        # Should only keep detections with conf >= 0.5 (indices 0, 1, 4)
        self.assertEqual(len(result), 3)
        expected_confidences = [0.8, 0.6, 0.7]
        np.testing.assert_array_almost_equal(
            sorted(result.scores), sorted(expected_confidences)
        )
        print("Confidence filtering test passed")

    def test_empty_detections(self):
        """Test handling of empty detections"""
        postprocessor = YOLOv8DetPostprocessor(self.config_without_nms)

        # Create detection output with low confidence scores
        detection_output = np.random.rand(1, 5, 6) * 0.1  # All low confidence
        raw_outputs = {"output": detection_output}

        result = postprocessor.postprocess(raw_outputs)

        self.assertEqual(len(result), 0)
        self.assertEqual(result.boxes.shape[0], 0)
        print("Empty detections test passed")

    def test_invalid_format_handling(self):
        """Test handling of invalid input formats"""
        postprocessor = YOLOv8DetPostprocessor(self.config_without_nms)

        # Test wrong number of features for non-NMS format
        invalid_output = np.random.rand(1, 5, 5)  # Should be 6 features
        raw_outputs = {"output": invalid_output}

        with self.assertRaises(RuntimeError):
            postprocessor.postprocess(raw_outputs)

        # Test wrong dimensions
        invalid_output_2d = np.random.rand(5, 6)  # Should be 3D for non-NMS
        raw_outputs = {"output": invalid_output_2d}

        with self.assertRaises(RuntimeError):
            postprocessor.postprocess(raw_outputs)

        print("Invalid format handling test passed")


def create_demo_usage():
    """Demonstrate usage of the updated postprocessor"""
    print("\n=== YOLOv8 Detection Postprocessor Demo ===")

    # Demo 1: Non-NMS format
    print("\n1. Non-NMS Format Demo:")
    config_no_nms = PostprocessConfig(
        num_classes=80,
        det_conf_threshold=0.3,
        det_max_detections=50,
        nms=False,
        class_names=[f"class_{i}" for i in range(80)],
    )

    postprocessor_no_nms = YOLOv8DetPostprocessor(config_no_nms)

    # Create sample non-NMS output
    sample_output = np.array(
        [
            [
                [100, 100, 200, 200, 0.85, 0],
                [300, 150, 450, 300, 0.72, 1],
                [50, 400, 180, 550, 0.91, 2],
            ]
        ]
    )

    result = postprocessor_no_nms.postprocess({"output": sample_output})
    print(f"Detected {len(result)} objects:")
    for i in range(len(result)):
        box = result.boxes[i]
        score = result.scores[i]
        class_id = result.class_ids[i]
        class_name = postprocessor_no_nms.get_class_name(class_id)
        print(
            f"  Object {i+1}: {class_name} (conf: {score:.3f}) at [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
        )

    # Demo 2: NMS format with multiple outputs
    print("\n2. NMS Format Demo (Multiple Outputs):")
    config_nms = PostprocessConfig(
        num_classes=80,
        det_conf_threshold=0.25,
        nms_iou_threshold=0.5,
        det_max_detections=100,
        nms=True,
        det_class_agnostic=False,
    )

    postprocessor_nms = YOLOv8DetPostprocessor(config_nms)

    # Create sample multi-head outputs (traditional YOLOv8 format)
    np.random.seed(123)
    head1 = np.random.rand(1, 3, 84)  # 4 + 80 classes
    head2 = np.random.rand(1, 4, 84)
    head3 = np.random.rand(1, 2, 84)

    # Set some realistic values
    for head in [head1, head2, head3]:
        head[0, :, :4] = np.random.uniform(100, 500, (head.shape[1], 4))  # coordinates
        head[0, :, 4:6] = 0.8  # High confidence for first 2 classes

    multi_outputs = {
        "detection_head_small": head1,
        "detection_head_medium": head2,
        "detection_head_large": head3,
    }

    result_nms = postprocessor_nms.postprocess(multi_outputs)
    print(
        f"After NMS: Detected {len(result_nms)} objects from {len(multi_outputs)} detection heads"
    )


if __name__ == "__main__":
    # Run tests
    unittest.main(argv=[""], exit=False, verbosity=2)

    # Run demo
    create_demo_usage()
