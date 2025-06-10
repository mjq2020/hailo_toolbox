"""
YOLOv8 Keypoint Detection Postprocessor

This module implements postprocessing for YOLOv8 pose estimation models.
It handles keypoint detection, pose estimation, and person detection with proper NMS and coordinate scaling.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging

from .base import (
    BasePostprocessor,
    PostprocessConfig,
    KeypointResult,
    DetectionResult,
    non_max_suppression,
    scale_boxes,
    scale_keypoints,
)


logger = logging.getLogger(__name__)


# COCO keypoint names and connections for visualization
COCO_KEYPOINT_NAMES = [
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
]

COCO_KEYPOINT_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # Head
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # Arms
    (5, 11),
    (6, 12),
    (11, 12),  # Torso
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # Legs
]


class YOLOv8KpPostprocessor(BasePostprocessor):
    """
    Postprocessor for YOLOv8 keypoint detection (pose estimation) models.

    This class handles the conversion of raw YOLOv8 keypoint outputs into
    structured keypoint results with proper NMS, coordinate scaling, and pose validation.

    Supported output formats:
    - YOLOv8-pose: (batch_size, num_detections, 4 + 1 + num_keypoints * 3)
    - Format: [x, y, w, h, person_conf, kp1_x, kp1_y, kp1_vis, kp2_x, kp2_y, kp2_vis, ...]
    """

    def __init__(self, config: Optional[PostprocessConfig] = None):
        """
        Initialize the YOLOv8 keypoint postprocessor.

        Args:
            config: Postprocessing configuration. If None, default config is used.
        """
        super().__init__(config)

        # Validate configuration
        self._validate_config()

        # Set up keypoint names and connections if not provided
        if self.config.keypoint_names is None:
            self.config.keypoint_names = COCO_KEYPOINT_NAMES[
                : self.config.num_keypoints
            ]

        if self.config.keypoint_connections is None:
            self.config.keypoint_connections = COCO_KEYPOINT_CONNECTIONS

        logger.info(
            f"Initialized YOLOv8KpPostprocessor with {self.config.num_keypoints} keypoints"
        )

    def _validate_config(self) -> None:
        """
        Validate the postprocessing configuration.

        Raises:
            ValueError: If configuration parameters are invalid
        """
        if self.config.kp_conf_threshold < 0 or self.config.kp_conf_threshold > 1:
            raise ValueError(
                f"Keypoint confidence threshold must be between 0 and 1, "
                f"got {self.config.kp_conf_threshold}"
            )

        if (
            self.config.kp_visibility_threshold < 0
            or self.config.kp_visibility_threshold > 1
        ):
            raise ValueError(
                f"Keypoint visibility threshold must be between 0 and 1, "
                f"got {self.config.kp_visibility_threshold}"
            )

        if self.config.nms_iou_threshold < 0 or self.config.nms_iou_threshold > 1:
            raise ValueError(
                f"NMS IoU threshold must be between 0 and 1, "
                f"got {self.config.nms_iou_threshold}"
            )

        if self.config.num_keypoints <= 0:
            raise ValueError(
                f"Number of keypoints must be positive, got {self.config.num_keypoints}"
            )

        if self.config.kp_max_persons <= 0:
            raise ValueError(
                f"Maximum persons must be positive, got {self.config.kp_max_persons}"
            )

    def postprocess(
        self,
        raw_outputs: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> KeypointResult:
        """
        Postprocess raw YOLOv8 keypoint detection outputs.

        Args:
            raw_outputs: Dictionary containing raw model outputs
            original_shape: Original image shape (height, width) for coordinate scaling

        Returns:
            KeypointResult containing processed keypoints and person detections

        Raises:
            ValueError: If output format is not supported
            RuntimeError: If postprocessing fails
        """
        try:
            # Extract the main keypoint output
            keypoint_output = self._extract_keypoint_output(raw_outputs)

            # Handle batch dimension
            if len(keypoint_output.shape) == 3:
                # Process first batch only
                keypoint_output = keypoint_output[0]

            # Parse keypoints and person detections
            boxes, scores, keypoints = self._parse_keypoints(keypoint_output)

            # Apply confidence filtering
            valid_detections = scores >= self.config.kp_conf_threshold
            if not np.any(valid_detections):
                logger.debug("No person detections above confidence threshold")
                return self._create_empty_result()

            boxes = boxes[valid_detections]
            scores = scores[valid_detections]
            keypoints = keypoints[valid_detections]

            # Filter keypoints by visibility threshold
            keypoints = self._filter_keypoints_by_visibility(keypoints)

            # Apply Non-Maximum Suppression if enabled
            if self.config.nms:
                keep_indices = self._apply_nms(boxes, scores)
                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                keypoints = keypoints[keep_indices]

            # Limit number of persons
            if len(boxes) > self.config.kp_max_persons:
                # Sort by confidence and keep top persons
                top_indices = np.argsort(scores)[::-1][: self.config.kp_max_persons]
                boxes = boxes[top_indices]
                scores = scores[top_indices]
                keypoints = keypoints[top_indices]

            # Scale results to original image size if needed
            if original_shape is not None:
                boxes = scale_boxes(boxes, original_shape, self.config.input_shape)
                keypoints = scale_keypoints(
                    keypoints, original_shape, self.config.input_shape
                )
                boxes = self._clip_boxes(boxes, original_shape)
                keypoints = self._clip_keypoints(keypoints, original_shape)

            # Validate poses
            keypoints = self._validate_poses(keypoints)

            logger.debug(f"Postprocessed {len(boxes)} person keypoint detections")

            return KeypointResult(keypoints=keypoints, scores=scores, boxes=boxes)

        except Exception as e:
            logger.error(f"Error in keypoint postprocessing: {str(e)}")
            raise RuntimeError(f"Keypoint postprocessing failed: {str(e)}") from e

    def _extract_keypoint_output(
        self, raw_outputs: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Extract the main keypoint output from raw outputs.

        Args:
            raw_outputs: Dictionary of raw model outputs

        Returns:
            Main keypoint output array

        Raises:
            ValueError: If no suitable output is found
        """
        # Try common output names
        common_names = ["output", "output0", "keypoints", "poses", "predictions"]

        for name in common_names:
            if name in raw_outputs:
                return raw_outputs[name]

        # If no common name found, use the first output
        if len(raw_outputs) == 1:
            return list(raw_outputs.values())[0]

        # If multiple outputs, try to find the largest one (likely keypoints)
        largest_output = None
        largest_size = 0

        for name, output in raw_outputs.items():
            if output.size > largest_size:
                largest_size = output.size
                largest_output = output

        if largest_output is not None:
            logger.warning(f"Using largest output for keypoint postprocessing")
            return largest_output

        raise ValueError("Could not identify keypoint output in raw_outputs")

    def _parse_keypoints(
        self, keypoint_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse keypoint output into boxes, scores, and keypoints.

        Args:
            keypoint_output: Raw keypoint output array

        Returns:
            Tuple of (boxes, scores, keypoints)

        Raises:
            ValueError: If output format is not supported
        """
        if len(keypoint_output.shape) != 2:
            raise ValueError(
                f"Expected 2D keypoint output, got shape {keypoint_output.shape}"
            )

        num_detections, num_features = keypoint_output.shape

        # Expected format: [x, y, w, h, person_conf, kp1_x, kp1_y, kp1_vis, kp2_x, kp2_y, kp2_vis, ...]
        expected_features = (
            5 + self.config.num_keypoints * 3
        )  # 4 box + 1 conf + keypoints * 3

        if num_features != expected_features:
            # Try to infer the number of keypoints from the output size
            if num_features >= 5 and (num_features - 5) % 3 == 0:
                inferred_keypoints = (num_features - 5) // 3
                logger.warning(
                    f"Inferred {inferred_keypoints} keypoints from output shape, "
                    f"expected {self.config.num_keypoints}"
                )
                self.config.num_keypoints = inferred_keypoints
            else:
                raise ValueError(
                    f"Unsupported keypoint output format. "
                    f"Expected {expected_features} features, got {num_features}"
                )

        # Parse components
        boxes_xywh = keypoint_output[:, :4]
        person_scores = keypoint_output[:, 4]
        keypoint_data = keypoint_output[:, 5:].reshape(
            num_detections, self.config.num_keypoints, 3
        )

        # Convert boxes from xywh to xyxy format
        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)

        return boxes_xyxy, person_scores, keypoint_data

    def _xywh_to_xyxy(self, boxes_xywh: np.ndarray) -> np.ndarray:
        """
        Convert bounding boxes from xywh to xyxy format.

        Args:
            boxes_xywh: Boxes in xywh format (center_x, center_y, width, height)

        Returns:
            Boxes in xyxy format (x1, y1, x2, y2)
        """
        boxes_xyxy = np.zeros_like(boxes_xywh)

        # Calculate corners
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        return boxes_xyxy

    def _filter_keypoints_by_visibility(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Filter keypoints based on visibility threshold.

        Args:
            keypoints: Keypoints with shape (num_persons, num_keypoints, 3)

        Returns:
            Filtered keypoints with low-visibility keypoints set to zero
        """
        if keypoints.size == 0:
            return keypoints

        # Create a copy to avoid modifying the original
        filtered_keypoints = keypoints.copy()

        # Set keypoints with low visibility to zero
        low_visibility_mask = keypoints[:, :, 2] < self.config.kp_visibility_threshold
        filtered_keypoints[low_visibility_mask] = 0.0

        return filtered_keypoints

    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """
        Apply Non-Maximum Suppression to person detections.

        Args:
            boxes: Bounding boxes in xyxy format
            scores: Confidence scores

        Returns:
            Indices of detections to keep
        """
        return non_max_suppression(
            boxes=boxes,
            scores=scores,
            iou_threshold=self.config.nms_iou_threshold,
            score_threshold=self.config.kp_conf_threshold,
            max_detections=self.config.kp_max_persons,
        )

    def _clip_boxes(
        self, boxes: np.ndarray, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Clip bounding boxes to image boundaries.

        Args:
            boxes: Bounding boxes in xyxy format
            image_shape: Image shape (height, width)

        Returns:
            Clipped bounding boxes
        """
        if boxes.size == 0:
            return boxes

        height, width = image_shape

        # Clip coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)  # x1
        boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)  # y1
        boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)  # x2
        boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)  # y2

        # Ensure x2 > x1 and y2 > y1
        boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1)
        boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1)

        return boxes

    def _clip_keypoints(
        self, keypoints: np.ndarray, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Clip keypoints to image boundaries.

        Args:
            keypoints: Keypoints with shape (num_persons, num_keypoints, 3)
            image_shape: Image shape (height, width)

        Returns:
            Clipped keypoints
        """
        if keypoints.size == 0:
            return keypoints

        height, width = image_shape
        clipped_keypoints = keypoints.copy()

        # Clip x coordinates
        clipped_keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0, width - 1)

        # Clip y coordinates
        clipped_keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0, height - 1)

        # Keep visibility unchanged
        # clipped_keypoints[:, :, 2] remains the same

        return clipped_keypoints

    def _validate_poses(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Validate poses by checking keypoint consistency and anatomical constraints.

        Args:
            keypoints: Keypoints with shape (num_persons, num_keypoints, 3)

        Returns:
            Validated keypoints with invalid poses filtered
        """
        if keypoints.size == 0:
            return keypoints

        validated_keypoints = keypoints.copy()

        # Basic validation: check for reasonable keypoint positions
        for person_idx in range(len(keypoints)):
            person_keypoints = keypoints[person_idx]

            # Count visible keypoints
            visible_keypoints = np.sum(
                person_keypoints[:, 2] > self.config.kp_visibility_threshold
            )

            # Require minimum number of visible keypoints for a valid pose
            min_visible_keypoints = max(3, self.config.num_keypoints // 4)
            if visible_keypoints < min_visible_keypoints:
                # Mark all keypoints as invisible for this person
                validated_keypoints[person_idx, :, 2] = 0.0
                continue

            # Additional anatomical validation can be added here
            # For example, checking if head keypoints are above torso keypoints
            validated_keypoints[person_idx] = self._validate_single_pose(
                person_keypoints
            )

        return validated_keypoints

    def _validate_single_pose(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Validate a single pose for anatomical consistency.

        Args:
            keypoints: Single person keypoints with shape (num_keypoints, 3)

        Returns:
            Validated keypoints for the person
        """
        validated_keypoints = keypoints.copy()

        # Example validation: head should be above shoulders (with reasonable tolerance)
        if self.config.num_keypoints >= 6:  # Assuming COCO format
            # Nose (0), left_shoulder (5), right_shoulder (6)
            nose_visible = keypoints[0, 2] > self.config.kp_visibility_threshold
            left_shoulder_visible = (
                keypoints[5, 2] > self.config.kp_visibility_threshold
            )
            right_shoulder_visible = (
                keypoints[6, 2] > self.config.kp_visibility_threshold
            )

            if nose_visible and (left_shoulder_visible or right_shoulder_visible):
                nose_y = keypoints[0, 1]
                shoulder_y = np.mean(
                    [
                        keypoints[5, 1] if left_shoulder_visible else nose_y,
                        keypoints[6, 1] if right_shoulder_visible else nose_y,
                    ]
                )

                # Only mark as invalid if nose is significantly below shoulders
                # Use a more lenient threshold to avoid false negatives
                if nose_y > shoulder_y + 100:  # 100 pixel threshold (more lenient)
                    validated_keypoints[0, 2] = 0.0  # Hide nose

        return validated_keypoints

    def _create_empty_result(self) -> KeypointResult:
        """
        Create an empty keypoint result.

        Returns:
            Empty KeypointResult
        """
        return KeypointResult(
            keypoints=np.empty((0, self.config.num_keypoints, 3), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            boxes=np.empty((0, 4), dtype=np.float32),
        )

    def get_keypoint_name(self, keypoint_id: int) -> str:
        """
        Get keypoint name for a given keypoint ID.

        Args:
            keypoint_id: Keypoint ID

        Returns:
            Keypoint name
        """
        if 0 <= keypoint_id < len(self.config.keypoint_names):
            return self.config.keypoint_names[keypoint_id]
        else:
            return f"keypoint_{keypoint_id}"

    def filter_by_confidence(
        self, result: KeypointResult, min_confidence: float
    ) -> KeypointResult:
        """
        Filter keypoint results by minimum confidence threshold.

        Args:
            result: Keypoint result to filter
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered keypoint result
        """
        if len(result) == 0:
            return result

        # Create mask for confident detections
        conf_mask = result.scores >= min_confidence

        if not np.any(conf_mask):
            return self._create_empty_result()

        return KeypointResult(
            keypoints=result.keypoints[conf_mask],
            scores=result.scores[conf_mask],
            boxes=result.boxes[conf_mask] if result.boxes is not None else None,
        )

    def filter_by_keypoint_count(
        self, result: KeypointResult, min_keypoints: int
    ) -> KeypointResult:
        """
        Filter keypoint results by minimum number of visible keypoints.

        Args:
            result: Keypoint result to filter
            min_keypoints: Minimum number of visible keypoints required

        Returns:
            Filtered keypoint result
        """
        if len(result) == 0:
            return result

        # Count visible keypoints for each person
        visible_counts = np.sum(
            result.keypoints[:, :, 2] > self.config.kp_visibility_threshold, axis=1
        )

        # Create mask for persons with enough visible keypoints
        keypoint_mask = visible_counts >= min_keypoints

        if not np.any(keypoint_mask):
            return self._create_empty_result()

        return KeypointResult(
            keypoints=result.keypoints[keypoint_mask],
            scores=result.scores[keypoint_mask],
            boxes=result.boxes[keypoint_mask] if result.boxes is not None else None,
        )

    def calculate_pose_similarity(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        """
        Calculate similarity between two poses using keypoint distances.

        Args:
            pose1: First pose keypoints with shape (num_keypoints, 3)
            pose2: Second pose keypoints with shape (num_keypoints, 3)

        Returns:
            Similarity score between 0 and 1 (1 = identical poses)
        """
        if pose1.shape != pose2.shape:
            return 0.0

        # Only consider visible keypoints in both poses
        visible_mask = (pose1[:, 2] > self.config.kp_visibility_threshold) & (
            pose2[:, 2] > self.config.kp_visibility_threshold
        )

        if not np.any(visible_mask):
            return 0.0

        # Calculate normalized distances between corresponding keypoints
        distances = np.sqrt(
            (pose1[visible_mask, 0] - pose2[visible_mask, 0]) ** 2
            + (pose1[visible_mask, 1] - pose2[visible_mask, 1]) ** 2
        )

        # Normalize by image size (assuming input shape)
        max_distance = np.sqrt(
            self.config.input_shape[0] ** 2 + self.config.input_shape[1] ** 2
        )
        normalized_distances = distances / max_distance

        # Convert to similarity (closer = more similar)
        similarities = 1.0 - np.clip(normalized_distances, 0, 1)

        return np.mean(similarities)
