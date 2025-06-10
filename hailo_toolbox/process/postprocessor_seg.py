"""
YOLOv8 Segmentation Postprocessor

This module implements postprocessing for YOLOv8 instance segmentation models.
It handles detection postprocessing and mask generation from prototype coefficients.
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional, Union
import logging

from .base import (
    BasePostprocessor,
    PostprocessConfig,
    DetectionResult,
    SegmentationResult,
    non_max_suppression,
    scale_boxes,
)


logger = logging.getLogger(__name__)


class YOLOv8SegPostprocessor(BasePostprocessor):
    """
    Postprocessor for YOLOv8 instance segmentation models.

    This class handles the conversion of raw YOLOv8 segmentation outputs into
    structured segmentation results with proper mask generation, NMS, and coordinate scaling.

    Supported output formats:
    - YOLOv8-seg: Detection output + mask prototypes + mask coefficients
    - Combined format: (batch_size, num_detections, 4 + num_classes + num_masks)
    """

    def __init__(self, config: Optional[PostprocessConfig] = None):
        """
        Initialize the YOLOv8 segmentation postprocessor.

        Args:
            config: Postprocessing configuration. If None, default config is used.
        """
        super().__init__(config)

        # Validate configuration
        self._validate_config()

        # Set up class names if not provided
        if self.config.class_names is None:
            self.config.class_names = [
                f"class_{i}" for i in range(self.config.num_classes)
            ]

        # Default number of mask prototypes (YOLOv8 typically uses 32)
        self.num_masks = 32

        logger.info(
            f"Initialized YOLOv8SegPostprocessor with {self.config.num_classes} classes"
        )

    def _validate_config(self) -> None:
        """
        Validate the postprocessing configuration.

        Raises:
            ValueError: If configuration parameters are invalid
        """
        if self.config.seg_conf_threshold < 0 or self.config.seg_conf_threshold > 1:
            raise ValueError(
                f"Segmentation confidence threshold must be between 0 and 1, "
                f"got {self.config.seg_conf_threshold}"
            )

        if self.config.seg_mask_threshold < 0 or self.config.seg_mask_threshold > 1:
            raise ValueError(
                f"Mask threshold must be between 0 and 1, "
                f"got {self.config.seg_mask_threshold}"
            )

        if self.config.nms_iou_threshold < 0 or self.config.nms_iou_threshold > 1:
            raise ValueError(
                f"NMS IoU threshold must be between 0 and 1, "
                f"got {self.config.nms_iou_threshold}"
            )

        if self.config.num_classes <= 0:
            raise ValueError(
                f"Number of classes must be positive, got {self.config.num_classes}"
            )

        if self.config.seg_max_instances <= 0:
            raise ValueError(
                f"Maximum instances must be positive, got {self.config.seg_max_instances}"
            )

    def postprocess(
        self,
        raw_outputs: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> SegmentationResult:
        """
        Postprocess raw YOLOv8 segmentation outputs.

        Args:
            raw_outputs: Dictionary containing raw model outputs
            original_shape: Original image shape (height, width) for coordinate scaling

        Returns:
            SegmentationResult containing processed segmentation masks and detections

        Raises:
            ValueError: If output format is not supported
            RuntimeError: If postprocessing fails
        """
        try:
            # Extract outputs
            detection_output, mask_prototypes = self._extract_outputs(raw_outputs)

            # Handle batch dimension
            if len(detection_output.shape) == 3:
                detection_output = detection_output[0]
            if len(mask_prototypes.shape) == 4:
                mask_prototypes = mask_prototypes[0]

            # Parse detections and mask coefficients
            boxes, scores, class_ids, mask_coeffs = self._parse_detections(
                detection_output
            )

            # Apply confidence filtering
            valid_detections = scores >= self.config.seg_conf_threshold
            if not np.any(valid_detections):
                logger.debug("No detections above confidence threshold")
                return self._create_empty_result()

            boxes = boxes[valid_detections]
            scores = scores[valid_detections]
            class_ids = class_ids[valid_detections]
            mask_coeffs = mask_coeffs[valid_detections]

            # Apply Non-Maximum Suppression if enabled
            if self.config.nms:
                keep_indices = self._apply_nms(boxes, scores, class_ids)
                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                class_ids = class_ids[keep_indices]
                mask_coeffs = mask_coeffs[keep_indices]

            # Limit number of instances
            if len(boxes) > self.config.seg_max_instances:
                # Sort by confidence and keep top instances
                top_indices = np.argsort(scores)[::-1][: self.config.seg_max_instances]
                boxes = boxes[top_indices]
                scores = scores[top_indices]
                class_ids = class_ids[top_indices]
                mask_coeffs = mask_coeffs[top_indices]

            # Generate masks from prototypes and coefficients
            masks = self._generate_masks(mask_prototypes, mask_coeffs, boxes)

            # Scale results to original image size if needed
            if original_shape is not None:
                boxes = scale_boxes(boxes, original_shape, self.config.input_shape)
                masks = self._scale_masks(masks, original_shape)
                boxes = self._clip_boxes(boxes, original_shape)

            logger.debug(f"Postprocessed {len(boxes)} segmentation instances")

            return SegmentationResult(
                masks=masks, scores=scores, class_ids=class_ids, boxes=boxes
            )

        except Exception as e:
            logger.error(f"Error in segmentation postprocessing: {str(e)}")
            raise RuntimeError(f"Segmentation postprocessing failed: {str(e)}") from e

    def _extract_outputs(
        self, raw_outputs: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract detection and mask prototype outputs from raw outputs.

        Args:
            raw_outputs: Dictionary of raw model outputs

        Returns:
            Tuple of (detection_output, mask_prototypes)

        Raises:
            ValueError: If required outputs are not found
        """
        detection_output = None
        mask_prototypes = None

        # Try to find detection output
        detection_names = ["output0", "detections", "predictions", "boxes"]
        for name in detection_names:
            if name in raw_outputs:
                detection_output = raw_outputs[name]
                break

        # Try to find mask prototypes
        prototype_names = ["output1", "prototypes", "masks", "proto"]
        for name in prototype_names:
            if name in raw_outputs:
                mask_prototypes = raw_outputs[name]
                break

        # If not found by name, try to infer from shapes
        if detection_output is None or mask_prototypes is None:
            outputs_by_size = sorted(
                raw_outputs.items(), key=lambda x: x[1].size, reverse=True
            )

            if len(outputs_by_size) >= 2:
                # Largest output is likely detections
                if detection_output is None:
                    detection_output = outputs_by_size[0][1]

                # Look for mask prototypes (should have spatial dimensions)
                for name, output in outputs_by_size[1:]:
                    if len(output.shape) >= 3 and mask_prototypes is None:
                        mask_prototypes = output
                        break

        if detection_output is None:
            raise ValueError("Could not find detection output in raw_outputs")

        if mask_prototypes is None:
            raise ValueError("Could not find mask prototypes in raw_outputs")

        return detection_output, mask_prototypes

    def _parse_detections(
        self, detection_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse detection output into boxes, scores, class IDs, and mask coefficients.

        Args:
            detection_output: Raw detection output array

        Returns:
            Tuple of (boxes, scores, class_ids, mask_coefficients)

        Raises:
            ValueError: If output format is not supported
        """
        if len(detection_output.shape) != 2:
            raise ValueError(
                f"Expected 2D detection output, got shape {detection_output.shape}"
            )

        num_detections, num_features = detection_output.shape

        # Expected format: [x, y, w, h, class_0, class_1, ..., mask_coeff_0, mask_coeff_1, ...]
        expected_features = 4 + self.config.num_classes + self.num_masks

        if num_features < 4 + self.config.num_classes:
            raise ValueError(
                f"Insufficient features in detection output. "
                f"Expected at least {4 + self.config.num_classes}, got {num_features}"
            )

        # Adjust num_masks based on actual output size
        actual_mask_coeffs = num_features - 4 - self.config.num_classes
        if actual_mask_coeffs > 0:
            self.num_masks = actual_mask_coeffs
        else:
            raise ValueError("No mask coefficients found in detection output")

        # Parse components
        boxes_xywh = detection_output[:, :4]
        class_scores = detection_output[:, 4 : 4 + self.config.num_classes]
        mask_coeffs = detection_output[:, 4 + self.config.num_classes :]

        # Get best class for each detection
        class_ids = np.argmax(class_scores, axis=1)
        scores = np.max(class_scores, axis=1)

        # Convert boxes from xywh to xyxy format
        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)

        return boxes_xyxy, scores, class_ids, mask_coeffs

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

    def _apply_nms(
        self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray
    ) -> np.ndarray:
        """
        Apply Non-Maximum Suppression to detections.

        Args:
            boxes: Bounding boxes in xyxy format
            scores: Confidence scores
            class_ids: Class IDs

        Returns:
            Indices of detections to keep
        """
        # Apply NMS per class for segmentation
        keep_indices = []

        for class_id in np.unique(class_ids):
            class_mask = class_ids == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]

            if len(class_boxes) == 0:
                continue

            class_keep = non_max_suppression(
                boxes=class_boxes,
                scores=class_scores,
                iou_threshold=self.config.nms_iou_threshold,
                score_threshold=self.config.seg_conf_threshold,
                max_detections=self.config.seg_max_instances,
            )

            # Convert back to original indices
            original_indices = np.where(class_mask)[0]
            keep_indices.extend(original_indices[class_keep])

        return np.array(keep_indices, dtype=np.int32)

    def _generate_masks(
        self, prototypes: np.ndarray, coefficients: np.ndarray, boxes: np.ndarray
    ) -> np.ndarray:
        """
        Generate instance masks from prototypes and coefficients.

        Args:
            prototypes: Mask prototypes with shape (num_masks, proto_h, proto_w)
            coefficients: Mask coefficients with shape (num_instances, num_masks)
            boxes: Bounding boxes in xyxy format for cropping masks

        Returns:
            Instance masks with shape (num_instances, mask_h, mask_w)
        """
        if len(coefficients) == 0:
            return np.empty(
                (0, self.config.input_shape[0], self.config.input_shape[1]),
                dtype=np.float32,
            )

        # Ensure prototypes have correct shape
        if len(prototypes.shape) == 3:
            num_masks, proto_h, proto_w = prototypes.shape
        else:
            raise ValueError(f"Expected 3D prototypes, got shape {prototypes.shape}")

        # Generate masks by matrix multiplication
        # prototypes: (num_masks, proto_h, proto_w)
        # coefficients: (num_instances, num_masks)
        # result: (num_instances, proto_h, proto_w)

        # Reshape prototypes for matrix multiplication
        prototypes_flat = prototypes.reshape(
            num_masks, -1
        )  # (num_masks, proto_h * proto_w)

        # Matrix multiplication: (num_instances, num_masks) @ (num_masks, proto_h * proto_w)
        masks_flat = (
            coefficients @ prototypes_flat
        )  # (num_instances, proto_h * proto_w)

        # Reshape back to spatial dimensions
        masks = masks_flat.reshape(
            -1, proto_h, proto_w
        )  # (num_instances, proto_h, proto_w)

        # Apply sigmoid activation
        masks = self._sigmoid(masks)

        # Resize masks to input shape if needed
        if (
            proto_h != self.config.input_shape[0]
            or proto_w != self.config.input_shape[1]
        ):
            masks = self._resize_masks(masks, self.config.input_shape)

        # Apply mask threshold
        masks = (masks > self.config.seg_mask_threshold).astype(np.float32)

        # Crop masks using bounding boxes
        masks = self._crop_masks(masks, boxes)

        return masks

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid activation function.

        Args:
            x: Input array

        Returns:
            Sigmoid activated array
        """
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def _resize_masks(
        self, masks: np.ndarray, target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize masks to target shape.

        Args:
            masks: Input masks with shape (num_instances, h, w)
            target_shape: Target shape (height, width)

        Returns:
            Resized masks
        """
        if len(masks) == 0:
            return masks

        target_h, target_w = target_shape
        resized_masks = []

        for mask in masks:
            resized_mask = cv2.resize(
                mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
            resized_masks.append(resized_mask)

        return np.array(resized_masks)

    def _crop_masks(self, masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Crop masks using bounding boxes to remove background.

        Args:
            masks: Instance masks with shape (num_instances, h, w)
            boxes: Bounding boxes in xyxy format

        Returns:
            Cropped masks
        """
        if len(masks) == 0 or len(boxes) == 0:
            return masks

        h, w = masks.shape[1], masks.shape[2]
        cropped_masks = masks.copy()

        for i, (mask, box) in enumerate(zip(masks, boxes)):
            # Convert box coordinates to integers
            x1, y1, x2, y2 = box.astype(int)

            # Clip coordinates to image boundaries
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # Create a mask for the bounding box region
            box_mask = np.zeros_like(mask)
            box_mask[y1:y2, x1:x2] = 1.0

            # Apply the box mask
            cropped_masks[i] = mask * box_mask

        return cropped_masks

    def _scale_masks(
        self, masks: np.ndarray, original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Scale masks to original image size.

        Args:
            masks: Masks with shape (num_instances, h, w)
            original_shape: Original image shape (height, width)

        Returns:
            Scaled masks
        """
        if len(masks) == 0:
            return masks

        return self._resize_masks(masks, original_shape)

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

    def _create_empty_result(self) -> SegmentationResult:
        """
        Create an empty segmentation result.

        Returns:
            Empty SegmentationResult
        """
        return SegmentationResult(
            masks=np.empty(
                (0, self.config.input_shape[0], self.config.input_shape[1]),
                dtype=np.float32,
            ),
            scores=np.empty((0,), dtype=np.float32),
            class_ids=np.empty((0,), dtype=np.int32),
            boxes=np.empty((0, 4), dtype=np.float32),
        )

    def get_class_name(self, class_id: int) -> str:
        """
        Get class name for a given class ID.

        Args:
            class_id: Class ID

        Returns:
            Class name
        """
        if 0 <= class_id < len(self.config.class_names):
            return self.config.class_names[class_id]
        else:
            return f"unknown_{class_id}"

    def filter_by_classes(
        self, result: SegmentationResult, class_ids: List[int]
    ) -> SegmentationResult:
        """
        Filter segmentation results by specific class IDs.

        Args:
            result: Segmentation result to filter
            class_ids: List of class IDs to keep

        Returns:
            Filtered segmentation result
        """
        if len(result) == 0:
            return result

        # Create mask for desired classes
        class_mask = np.isin(result.class_ids, class_ids)

        if not np.any(class_mask):
            return self._create_empty_result()

        return SegmentationResult(
            masks=result.masks[class_mask],
            scores=result.scores[class_mask] if result.scores is not None else None,
            class_ids=(
                result.class_ids[class_mask] if result.class_ids is not None else None
            ),
            boxes=result.boxes[class_mask] if result.boxes is not None else None,
        )

    def filter_by_confidence(
        self, result: SegmentationResult, min_confidence: float
    ) -> SegmentationResult:
        """
        Filter segmentation results by minimum confidence threshold.

        Args:
            result: Segmentation result to filter
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered segmentation result
        """
        if len(result) == 0 or result.scores is None:
            return result

        # Create mask for confident detections
        conf_mask = result.scores >= min_confidence

        if not np.any(conf_mask):
            return self._create_empty_result()

        return SegmentationResult(
            masks=result.masks[conf_mask],
            scores=result.scores[conf_mask],
            class_ids=(
                result.class_ids[conf_mask] if result.class_ids is not None else None
            ),
            boxes=result.boxes[conf_mask] if result.boxes is not None else None,
        )
