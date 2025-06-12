"""
Unit tests for YOLOv8 postprocessors.

This module contains comprehensive tests for detection, segmentation, and keypoint postprocessors
to ensure they handle various input formats correctly and produce valid outputs.
"""

import unittest
import numpy as np
from typing import Dict

from .base import PostprocessConfig
from .postprocessor.postprocessor_det import YOLOv8DetPostprocessor
from .postprocessor.postprocessor_seg import YOLOv8SegPostprocessor
from .postprocessor_kp import YOLOv8KpPostprocessor


class TestYOLOv8DetPostprocessor(unittest.TestCase):
    """Test cases for YOLOv8 detection postprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = PostprocessConfig(
            num_classes=80,
            det_conf_threshold=0.25,
            nms_iou_threshold=0.5,
            det_max_detections=100,
            input_shape=(640, 640),
        )
        self.postprocessor = YOLOv8DetPostprocessor(self.config)

    def test_initialization(self):
        """Test postprocessor initialization."""
        self.assertEqual(self.postprocessor.config.num_classes, 80)
        self.assertEqual(len(self.postprocessor.config.class_names), 80)
        self.assertTrue(self.postprocessor.config.nms)

    def test_detection_postprocessing_standard_format(self):
        """Test detection postprocessing with standard YOLOv8 format."""
        # Create mock detection output: [x, y, w, h, class_0, class_1, ...]
        num_detections = 10
        num_features = 4 + self.config.num_classes

        # Generate random detections
        np.random.seed(42)
        detection_output = np.random.rand(num_detections, num_features)

        # Set some boxes to have high confidence
        detection_output[:5, 4:6] = 0.8  # High confidence for first 5 detections
        detection_output[5:, 4:] = 0.1  # Low confidence for rest

        raw_outputs = {"output": detection_output}

        result = self.postprocessor.postprocess(raw_outputs)

        # Check result structure
        self.assertIsNotNone(result)
        self.assertTrue(len(result) > 0)
        self.assertEqual(result.boxes.shape[1], 4)  # x1, y1, x2, y2
        self.assertEqual(len(result.scores), len(result.boxes))
        self.assertEqual(len(result.class_ids), len(result.boxes))

    def test_detection_postprocessing_with_objectness(self):
        """Test detection postprocessing with objectness score."""
        # Create mock detection output: [x, y, w, h, objectness, class_0, class_1, ...]
        num_detections = 5
        num_features = 5 + self.config.num_classes

        np.random.seed(42)
        detection_output = np.random.rand(num_detections, num_features)

        # Set objectness and class scores
        detection_output[:, 4] = 0.9  # High objectness
        detection_output[:, 5:7] = 0.8  # High class scores

        raw_outputs = {"output": detection_output}

        result = self.postprocessor.postprocess(raw_outputs)

        self.assertIsNotNone(result)
        self.assertTrue(len(result) > 0)

    def test_empty_detections(self):
        """Test handling of empty detections."""
        # Create detection output with low confidence scores
        num_detections = 5
        num_features = 4 + self.config.num_classes

        detection_output = (
            np.random.rand(num_detections, num_features) * 0.1
        )  # Low scores
        raw_outputs = {"output": detection_output}

        result = self.postprocessor.postprocess(raw_outputs)

        self.assertEqual(len(result), 0)
        self.assertEqual(result.boxes.shape[0], 0)

    def test_coordinate_scaling(self):
        """Test coordinate scaling to original image size."""
        num_detections = 3
        num_features = 4 + self.config.num_classes

        # Create detections with known coordinates
        detection_output = np.zeros((num_detections, num_features))
        detection_output[:, :4] = [
            [320, 320, 100, 100],  # Center box
            [160, 160, 50, 50],  # Top-left box
            [480, 480, 80, 80],
        ]  # Bottom-right box
        detection_output[:, 4] = 0.9  # High confidence

        raw_outputs = {"output": detection_output}
        original_shape = (1280, 1280)  # 2x scaling

        result = self.postprocessor.postprocess(raw_outputs, original_shape)

        # Check that coordinates are scaled
        self.assertTrue(np.all(result.boxes[:, 2] > result.boxes[:, 0]))  # x2 > x1
        self.assertTrue(np.all(result.boxes[:, 3] > result.boxes[:, 1]))  # y2 > y1
        self.assertTrue(np.max(result.boxes) <= 1280)  # Within image bounds

    def test_nms_filtering(self):
        """Test Non-Maximum Suppression filtering."""
        # Create overlapping detections
        num_detections = 4
        num_features = 4 + self.config.num_classes

        detection_output = np.zeros((num_detections, num_features))
        # Create overlapping boxes with different confidence scores
        detection_output[:, :4] = [
            [100, 100, 50, 50],  # High overlap
            [110, 110, 50, 50],  # High overlap
            [120, 120, 50, 50],  # High overlap
            [300, 300, 50, 50],
        ]  # Separate box

        # Set high confidence for the first class for all detections
        detection_output[:, 4] = [
            0.9,
            0.8,
            0.7,
            0.6,
        ]  # Decreasing confidence for class 0
        detection_output[:, 5:] = 0.1  # Low confidence for other classes

        raw_outputs = {"output": detection_output}

        result = self.postprocessor.postprocess(raw_outputs)

        # Should have fewer detections due to NMS (overlapping boxes should be filtered)
        # At minimum, we should have the separate box and the highest confidence overlapping box
        self.assertTrue(len(result) <= num_detections)
        self.assertTrue(len(result) >= 1)  # At least one detection should remain


class TestYOLOv8SegPostprocessor(unittest.TestCase):
    """Test cases for YOLOv8 segmentation postprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = PostprocessConfig(
            num_classes=80,
            seg_conf_threshold=0.25,
            seg_mask_threshold=0.5,
            nms_iou_threshold=0.5,
            seg_max_instances=100,
            input_shape=(640, 640),
        )
        self.postprocessor = YOLOv8SegPostprocessor(self.config)

    def test_initialization(self):
        """Test postprocessor initialization."""
        self.assertEqual(self.postprocessor.config.num_classes, 80)
        self.assertEqual(self.postprocessor.num_masks, 32)
        self.assertTrue(self.postprocessor.config.nms)

    def test_segmentation_postprocessing(self):
        """Test segmentation postprocessing with mock data."""
        # Create mock detection output with mask coefficients
        num_detections = 5
        num_mask_coeffs = 32
        num_features = 4 + self.config.num_classes + num_mask_coeffs

        np.random.seed(42)
        detection_output = np.random.rand(num_detections, num_features)
        detection_output[:, 4:6] = 0.8  # High class scores

        # Create mock mask prototypes
        proto_h, proto_w = 160, 160
        mask_prototypes = np.random.randn(num_mask_coeffs, proto_h, proto_w)

        raw_outputs = {"output0": detection_output, "output1": mask_prototypes}

        result = self.postprocessor.postprocess(raw_outputs)

        # Check result structure
        self.assertIsNotNone(result)
        if len(result) > 0:
            self.assertEqual(result.masks.shape[1:], self.config.input_shape)
            self.assertEqual(len(result.scores), len(result.masks))
            self.assertEqual(len(result.class_ids), len(result.masks))
            self.assertEqual(len(result.boxes), len(result.masks))

    def test_mask_generation(self):
        """Test mask generation from prototypes and coefficients."""
        # Create test data that matches the actual processing pipeline
        num_masks = 4
        proto_h, proto_w = 160, 160  # More realistic prototype size
        num_instances = 2

        # Create simple prototypes (identity-like patterns)
        prototypes = np.zeros((num_masks, proto_h, proto_w))
        prototypes[0, :80, :80] = 1.0  # Top-left quadrant
        prototypes[1, :80, 80:] = 1.0  # Top-right quadrant
        prototypes[2, 80:, :80] = 1.0  # Bottom-left quadrant
        prototypes[3, 80:, 80:] = 1.0  # Bottom-right quadrant

        # Create coefficients to select specific prototypes with very strong weights
        coefficients = np.array(
            [[5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0]]  # Select first prototype
        )  # Select fourth prototype

        # Create bounding boxes that cover reasonable areas of the image
        # Boxes should be in the coordinate space of the final mask size (640x640)
        boxes = np.array(
            [[100, 100, 300, 300], [400, 400, 600, 600]]  # Top-left area
        )  # Bottom-right area

        masks = self.postprocessor._generate_masks(prototypes, coefficients, boxes)

        self.assertEqual(masks.shape[0], num_instances)
        self.assertEqual(
            masks.shape[1:], self.config.input_shape
        )  # Should match input shape

        # Both masks should have some content (binary values)
        self.assertTrue(np.any(masks[0] > 0))  # First mask should have content
        self.assertTrue(np.any(masks[1] > 0))  # Second mask should have content

        # Masks should be binary (0 or 1) after thresholding
        unique_values_0 = np.unique(masks[0])
        unique_values_1 = np.unique(masks[1])
        self.assertTrue(all(val in [0.0, 1.0] for val in unique_values_0))
        self.assertTrue(all(val in [0.0, 1.0] for val in unique_values_1))

    def test_empty_segmentations(self):
        """Test handling of empty segmentations."""
        # Create segmentation output with low confidence scores
        num_detections = 3
        num_features = 4 + self.config.num_classes + 32

        detection_output = np.random.rand(num_detections, num_features) * 0.1
        mask_prototypes = np.random.randn(32, 160, 160)

        raw_outputs = {"output0": detection_output, "output1": mask_prototypes}

        result = self.postprocessor.postprocess(raw_outputs)

        self.assertEqual(len(result), 0)


class TestYOLOv8KpPostprocessor(unittest.TestCase):
    """Test cases for YOLOv8 keypoint postprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = PostprocessConfig(
            num_keypoints=17,  # COCO format
            kp_conf_threshold=0.25,
            kp_visibility_threshold=0.5,
            nms_iou_threshold=0.5,
            kp_max_persons=100,
            input_shape=(640, 640),
        )
        self.postprocessor = YOLOv8KpPostprocessor(self.config)

    def test_initialization(self):
        """Test postprocessor initialization."""
        self.assertEqual(self.postprocessor.config.num_keypoints, 17)
        self.assertEqual(len(self.postprocessor.config.keypoint_names), 17)
        self.assertTrue(self.postprocessor.config.nms)

    def test_keypoint_postprocessing(self):
        """Test keypoint postprocessing with mock data."""
        # Create mock keypoint output: [x, y, w, h, person_conf, kp1_x, kp1_y, kp1_vis, ...]
        num_persons = 3
        num_features = 5 + self.config.num_keypoints * 3

        np.random.seed(42)
        keypoint_output = np.random.rand(num_persons, num_features)

        # Set person confidence
        keypoint_output[:, 4] = [0.9, 0.8, 0.7]

        # Set some keypoints as visible
        for i in range(num_persons):
            kp_start = 5
            for j in range(self.config.num_keypoints):
                kp_idx = kp_start + j * 3
                keypoint_output[i, kp_idx : kp_idx + 2] = (
                    np.random.rand(2) * 640
                )  # x, y
                keypoint_output[i, kp_idx + 2] = np.random.rand()  # visibility

        raw_outputs = {"output": keypoint_output}

        result = self.postprocessor.postprocess(raw_outputs)

        # Check result structure
        self.assertIsNotNone(result)
        if len(result) > 0:
            self.assertEqual(result.keypoints.shape[1], self.config.num_keypoints)
            self.assertEqual(result.keypoints.shape[2], 3)  # x, y, visibility
            self.assertEqual(len(result.scores), len(result.keypoints))
            self.assertEqual(len(result.boxes), len(result.keypoints))

    def test_keypoint_visibility_filtering(self):
        """Test keypoint visibility filtering."""
        # Create keypoints with known visibility values
        keypoints = np.random.rand(2, 17, 3)
        keypoints[:, :, 2] = [0.3, 0.7] * 8 + [0.3]  # Alternating visibility

        filtered = self.postprocessor._filter_keypoints_by_visibility(keypoints)

        # Check that low visibility keypoints are set to zero
        low_vis_mask = keypoints[:, :, 2] < self.config.kp_visibility_threshold
        self.assertTrue(np.all(filtered[low_vis_mask] == 0.0))

    def test_pose_validation(self):
        """Test pose validation logic."""
        # Create a pose with reasonable keypoint positions
        keypoints = np.zeros((1, 17, 3))

        # Set nose and shoulders with proper anatomical positions (nose above shoulders)
        keypoints[0, 0] = [320, 200, 0.9]  # nose
        keypoints[0, 5] = [300, 300, 0.9]  # left shoulder (below nose)
        keypoints[0, 6] = [340, 300, 0.9]  # right shoulder (below nose)

        # Add some other visible keypoints to meet minimum requirements
        keypoints[0, 1] = [315, 195, 0.8]  # left eye
        keypoints[0, 2] = [325, 195, 0.8]  # right eye
        keypoints[0, 7] = [280, 350, 0.7]  # left elbow
        keypoints[0, 8] = [360, 350, 0.7]  # right elbow

        validated = self.postprocessor._validate_poses(keypoints)

        # With proper anatomical positions, nose should remain visible
        self.assertGreater(validated[0, 0, 2], 0)
        # Other keypoints should also remain visible
        self.assertGreater(validated[0, 5, 2], 0)  # left shoulder
        self.assertGreater(validated[0, 6, 2], 0)  # right shoulder

    def test_pose_similarity(self):
        """Test pose similarity calculation."""
        # Create two similar poses
        pose1 = np.random.rand(17, 3)
        pose1[:, 2] = 0.8  # High visibility

        pose2 = pose1.copy()
        pose2[:, :2] += np.random.randn(17, 2) * 5  # Small perturbation

        similarity = self.postprocessor.calculate_pose_similarity(pose1, pose2)

        self.assertGreater(similarity, 0.5)  # Should be reasonably similar
        self.assertLessEqual(similarity, 1.0)

    def test_empty_keypoints(self):
        """Test handling of empty keypoint detections."""
        # Create keypoint output with low confidence scores
        num_persons = 2
        num_features = 5 + self.config.num_keypoints * 3

        keypoint_output = np.random.rand(num_persons, num_features) * 0.1
        raw_outputs = {"output": keypoint_output}

        result = self.postprocessor.postprocess(raw_outputs)

        self.assertEqual(len(result), 0)


class TestPostprocessorIntegration(unittest.TestCase):
    """Integration tests for postprocessors."""

    def test_coordinate_scaling_consistency(self):
        """Test that all postprocessors handle coordinate scaling consistently."""
        config = PostprocessConfig(input_shape=(640, 640))
        original_shape = (1280, 1280)

        # Test detection postprocessor
        det_processor = YOLOv8DetPostprocessor(config)
        det_output = np.random.rand(5, 84)  # 4 + 80 classes
        det_output[:, 4:6] = 0.8
        det_result = det_processor.postprocess({"output": det_output}, original_shape)

        if len(det_result) > 0:
            self.assertTrue(np.all(det_result.boxes <= 1280))

        # Test keypoint postprocessor
        kp_processor = YOLOv8KpPostprocessor(config)
        kp_output = np.random.rand(3, 56)  # 5 + 17*3
        kp_output[:, 4] = 0.8
        kp_result = kp_processor.postprocess({"output": kp_output}, original_shape)

        if len(kp_result) > 0:
            self.assertTrue(np.all(kp_result.keypoints[:, :, :2] <= 1280))

    def test_config_validation(self):
        """Test configuration validation across all postprocessors."""
        # Test invalid confidence thresholds
        with self.assertRaises(ValueError):
            PostprocessConfig(det_conf_threshold=1.5)

        with self.assertRaises(ValueError):
            PostprocessConfig(seg_conf_threshold=-0.1)

        with self.assertRaises(ValueError):
            PostprocessConfig(kp_conf_threshold=2.0)

        # Test invalid class numbers
        with self.assertRaises(ValueError):
            YOLOv8DetPostprocessor(PostprocessConfig(num_classes=0))


if __name__ == "__main__":
    unittest.main()
