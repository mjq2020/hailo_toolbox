"""
Transform classes for image preprocessing operations.

This module provides a collection of transform classes that can be used
individually or chained together to create preprocessing pipelines.
Each transform follows a common interface for consistency and composability.
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, List, Dict, Any
from enum import Enum

from ..exceptions import ImageProcessingError, InvalidConfigError


class InterpolationMethod(Enum):
    """Enumeration of supported interpolation methods for resizing operations."""

    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA
    LANCZOS4 = cv2.INTER_LANCZOS4


class PaddingMode(Enum):
    """Enumeration of supported padding modes."""

    CONSTANT = cv2.BORDER_CONSTANT
    REFLECT = cv2.BORDER_REFLECT
    REFLECT_101 = cv2.BORDER_REFLECT_101
    REPLICATE = cv2.BORDER_REPLICATE
    WRAP = cv2.BORDER_WRAP


class BaseTransform(ABC):
    """
    Abstract base class for all image transforms.

    This class defines the common interface that all transforms must implement,
    ensuring consistency and enabling easy composition of transforms.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the base transform.

        Args:
            name (Optional[str]): Optional name for the transform
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the transform to an image.

        Args:
            image (np.ndarray): Input image as numpy array

        Returns:
            np.ndarray: Transformed image

        Raises:
            ImageProcessingError: If the transform fails
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration parameters of the transform.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        pass

    def validate_image(self, image: np.ndarray) -> None:
        """
        Validate input image format and properties.

        Args:
            image (np.ndarray): Input image to validate

        Raises:
            ImageProcessingError: If image validation fails
        """
        if not isinstance(image, np.ndarray):
            raise ImageProcessingError(
                f"Expected numpy array, got {type(image)}", operation=self.name
            )

        if image.size == 0:
            raise ImageProcessingError(
                "Empty image provided", operation=self.name, image_shape=image.shape
            )

        if len(image.shape) not in [2, 3]:
            raise ImageProcessingError(
                f"Expected 2D or 3D image, got {len(image.shape)}D",
                operation=self.name,
                image_shape=image.shape,
            )


class ResizeTransform(BaseTransform):
    """
    Transform for resizing images to specified dimensions.

    This transform supports various interpolation methods and can handle
    both fixed size resizing and aspect ratio preservation.
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        interpolation: Union[InterpolationMethod, str] = InterpolationMethod.LINEAR,
        preserve_aspect_ratio: bool = True,
        name: Optional[str] = None,
    ):
        """
        Initialize the resize transform.

        Args:
            target_size (Tuple[int, int]): Target size as (width, height)
            interpolation (Union[InterpolationMethod, str]): Interpolation method
            preserve_aspect_ratio (bool): Whether to preserve aspect ratio
            name (Optional[str]): Optional name for the transform

        Raises:
            InvalidConfigError: If configuration parameters are invalid
        """
        super().__init__(name)

        if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
            raise InvalidConfigError(
                "target_size must be a tuple of (width, height)",
                config_field="target_size",
                provided_value=target_size,
            )

        if target_size[0] <= 0 or target_size[1] <= 0:
            raise InvalidConfigError(
                "target_size dimensions must be positive",
                config_field="target_size",
                provided_value=target_size,
            )

        self.target_size = tuple(target_size)
        self.preserve_aspect_ratio = preserve_aspect_ratio

        # Handle interpolation method
        if isinstance(interpolation, str):
            try:
                self.interpolation = InterpolationMethod[interpolation.upper()]
            except KeyError:
                valid_methods = [method.name for method in InterpolationMethod]
                raise InvalidConfigError(
                    f"Invalid interpolation method. Valid options: {valid_methods}",
                    config_field="interpolation",
                    provided_value=interpolation,
                )
        else:
            self.interpolation = interpolation

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply resize transform to the image.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Resized image

        Raises:
            ImageProcessingError: If resizing fails
        """
        self.validate_image(image)

        try:
            if self.preserve_aspect_ratio:
                return self._resize_with_aspect_ratio(image)
            else:
                return cv2.resize(
                    image, self.target_size, interpolation=self.interpolation.value
                )
        except Exception as e:
            raise ImageProcessingError(
                f"Failed to resize image: {str(e)}",
                operation=self.name,
                image_shape=image.shape,
            )

    def _resize_with_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image while preserving aspect ratio.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Resized image with preserved aspect ratio
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)

        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image
        resized = cv2.resize(
            image, (new_w, new_h), interpolation=self.interpolation.value
        )

        # Add padding if necessary
        if new_w != target_w or new_h != target_h:
            # Create padded image
            if len(image.shape) == 3:
                padded = np.zeros(
                    (target_h, target_w, image.shape[2]), dtype=image.dtype
                )
            else:
                padded = np.zeros((target_h, target_w), dtype=image.dtype)

            # Calculate padding offsets
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2

            # Place resized image in center
            padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
            return padded

        return resized

    def get_config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        return {
            "target_size": self.target_size,
            "interpolation": self.interpolation.name,
            "preserve_aspect_ratio": self.preserve_aspect_ratio,
            "name": self.name,
        }


class NormalizationTransform(BaseTransform):
    """
    Transform for normalizing image pixel values.

    This transform supports various normalization strategies including
    standard normalization, min-max scaling, and custom mean/std normalization.
    """

    def __init__(
        self,
        mean: Union[float, List[float], np.ndarray] = 0.0,
        std: Union[float, List[float], np.ndarray] = 1.0,
        scale: float = 1.0,
        dtype: Optional[np.dtype] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize the normalization transform.

        Args:
            mean (Union[float, List[float], np.ndarray]): Mean values for normalization
            std (Union[float, List[float], np.ndarray]): Standard deviation values
            scale (float): Scaling factor applied before normalization
            dtype (Optional[np.dtype]): Target data type for output
            name (Optional[str]): Optional name for the transform
        """
        super().__init__(name)

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.scale = float(scale)
        self.dtype = dtype

        # Validate std values
        if np.any(self.std == 0):
            raise InvalidConfigError(
                "Standard deviation values cannot be zero",
                config_field="std",
                provided_value=std,
            )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply normalization transform to the image.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Normalized image

        Raises:
            ImageProcessingError: If normalization fails
        """
        self.validate_image(image)

        try:
            # Convert to float for processing
            normalized = image.astype(np.float32)

            # Apply scaling
            if self.scale != 1.0:
                normalized = normalized * self.scale

            # Apply normalization
            normalized = (normalized - self.mean) / self.std

            # Convert to target dtype if specified
            if self.dtype is not None:
                normalized = normalized.astype(self.dtype)

            return normalized

        except Exception as e:
            raise ImageProcessingError(
                f"Failed to normalize image: {str(e)}",
                operation=self.name,
                image_shape=image.shape,
            )

    def get_config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        return {
            "mean": self.mean.tolist() if self.mean.size > 1 else float(self.mean),
            "std": self.std.tolist() if self.std.size > 1 else float(self.std),
            "scale": self.scale,
            "dtype": str(self.dtype) if self.dtype else None,
            "name": self.name,
        }


class DataTypeTransform(BaseTransform):
    """
    Transform for converting image data types.

    This transform handles safe conversion between different numpy data types
    with optional value scaling and clipping.
    """

    def __init__(
        self,
        target_dtype: Union[np.dtype, str],
        scale_values: bool = True,
        clip_values: bool = True,
        name: Optional[str] = None,
    ):
        """
        Initialize the data type transform.

        Args:
            target_dtype (Union[np.dtype, str]): Target data type
            scale_values (bool): Whether to scale values when converting
            clip_values (bool): Whether to clip values to valid range
            name (Optional[str]): Optional name for the transform
        """
        super().__init__(name)

        if isinstance(target_dtype, str):
            try:
                self.target_dtype = np.dtype(target_dtype)
            except TypeError:
                raise InvalidConfigError(
                    f"Invalid data type: {target_dtype}",
                    config_field="target_dtype",
                    provided_value=target_dtype,
                )
        else:
            self.target_dtype = target_dtype

        self.scale_values = scale_values
        self.clip_values = clip_values

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data type conversion to the image.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Image with converted data type

        Raises:
            ImageProcessingError: If conversion fails
        """
        self.validate_image(image)

        try:
            if image.dtype == self.target_dtype:
                return image.copy()

            converted = image.astype(np.float64)

            if self.scale_values:
                # Scale from source range to target range
                source_info = (
                    np.iinfo(image.dtype)
                    if np.issubdtype(image.dtype, np.integer)
                    else None
                )
                target_info = (
                    np.iinfo(self.target_dtype)
                    if np.issubdtype(self.target_dtype, np.integer)
                    else None
                )

                if source_info and target_info:
                    # Integer to integer conversion
                    source_range = source_info.max - source_info.min
                    target_range = target_info.max - target_info.min

                    converted = (converted - source_info.min) / source_range
                    converted = converted * target_range + target_info.min
                elif target_info and not source_info:
                    # Float to integer conversion
                    converted = converted * target_info.max

            if self.clip_values and np.issubdtype(self.target_dtype, np.integer):
                info = np.iinfo(self.target_dtype)
                converted = np.clip(converted, info.min, info.max)

            return converted.astype(self.target_dtype)

        except Exception as e:
            raise ImageProcessingError(
                f"Failed to convert data type: {str(e)}",
                operation=self.name,
                image_shape=image.shape,
            )

    def get_config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        return {
            "target_dtype": str(self.target_dtype),
            "scale_values": self.scale_values,
            "clip_values": self.clip_values,
            "name": self.name,
        }


class PaddingTransform(BaseTransform):
    """
    Transform for adding padding to images.

    This transform supports various padding modes and can add padding
    to specific sides or all sides of an image.
    """

    def __init__(
        self,
        padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]],
        mode: Union[PaddingMode, str] = PaddingMode.CONSTANT,
        constant_value: Union[int, float, Tuple] = 0,
        name: Optional[str] = None,
    ):
        """
        Initialize the padding transform.

        Args:
            padding: Padding specification (all, (horizontal, vertical), or (top, right, bottom, left))
            mode (Union[PaddingMode, str]): Padding mode
            constant_value: Value to use for constant padding
            name (Optional[str]): Optional name for the transform
        """
        super().__init__(name)

        # Parse padding specification
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        elif len(padding) == 2:
            self.padding = (
                padding[1],
                padding[0],
                padding[1],
                padding[0],
            )  # (top, right, bottom, left)
        elif len(padding) == 4:
            self.padding = tuple(padding)
        else:
            raise InvalidConfigError(
                "padding must be int, (h, v), or (top, right, bottom, left)",
                config_field="padding",
                provided_value=padding,
            )

        # Handle padding mode
        if isinstance(mode, str):
            try:
                self.mode = PaddingMode[mode.upper()]
            except KeyError:
                valid_modes = [mode.name for mode in PaddingMode]
                raise InvalidConfigError(
                    f"Invalid padding mode. Valid options: {valid_modes}",
                    config_field="mode",
                    provided_value=mode,
                )
        else:
            self.mode = mode

        self.constant_value = constant_value

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply padding transform to the image.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Padded image

        Raises:
            ImageProcessingError: If padding fails
        """
        self.validate_image(image)

        try:
            top, right, bottom, left = self.padding

            return cv2.copyMakeBorder(
                image,
                top,
                bottom,
                left,
                right,
                self.mode.value,
                value=self.constant_value,
            )

        except Exception as e:
            raise ImageProcessingError(
                f"Failed to apply padding: {str(e)}",
                operation=self.name,
                image_shape=image.shape,
            )

    def get_config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        return {
            "padding": self.padding,
            "mode": self.mode.name,
            "constant_value": self.constant_value,
            "name": self.name,
        }


class CropTransform(BaseTransform):
    """
    Transform for cropping images to specified regions.

    This transform supports center cropping, corner cropping, and custom
    region cropping with various options for handling out-of-bounds regions.
    """

    def __init__(
        self,
        crop_size: Optional[Tuple[int, int]] = None,
        crop_region: Optional[Tuple[int, int, int, int]] = None,
        center_crop: bool = True,
        name: Optional[str] = None,
    ):
        """
        Initialize the crop transform.

        Args:
            crop_size (Optional[Tuple[int, int]]): Size of crop as (width, height)
            crop_region (Optional[Tuple[int, int, int, int]]): Specific region as (x, y, width, height)
            center_crop (bool): Whether to crop from center (if crop_size is used)
            name (Optional[str]): Optional name for the transform
        """
        super().__init__(name)

        if crop_size is None and crop_region is None:
            raise InvalidConfigError(
                "Either crop_size or crop_region must be specified",
                config_field="crop_size/crop_region",
            )

        if crop_size is not None and crop_region is not None:
            raise InvalidConfigError(
                "Cannot specify both crop_size and crop_region",
                config_field="crop_size/crop_region",
            )

        self.crop_size = crop_size
        self.crop_region = crop_region
        self.center_crop = center_crop

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply crop transform to the image.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Cropped image

        Raises:
            ImageProcessingError: If cropping fails
        """
        self.validate_image(image)

        try:
            h, w = image.shape[:2]

            if self.crop_region is not None:
                x, y, crop_w, crop_h = self.crop_region
                x2, y2 = x + crop_w, y + crop_h
            else:
                crop_w, crop_h = self.crop_size

                if self.center_crop:
                    x = (w - crop_w) // 2
                    y = (h - crop_h) // 2
                else:
                    x, y = 0, 0

                x2, y2 = x + crop_w, y + crop_h

            # Validate crop region
            if x < 0 or y < 0 or x2 > w or y2 > h:
                raise ImageProcessingError(
                    f"Crop region ({x}, {y}, {x2}, {y2}) exceeds image bounds ({w}, {h})",
                    operation=self.name,
                    image_shape=image.shape,
                )

            return image[y:y2, x:x2]

        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ImageProcessingError(
                f"Failed to crop image: {str(e)}",
                operation=self.name,
                image_shape=image.shape,
            )

    def get_config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        return {
            "crop_size": self.crop_size,
            "crop_region": self.crop_region,
            "center_crop": self.center_crop,
            "name": self.name,
        }
