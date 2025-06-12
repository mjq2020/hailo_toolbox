"""
USB/Integrated webcam source implementation.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from .base import BaseSource, SourceType


class WebcamSource(BaseSource):
    """
    Video source for USB webcams or integrated cameras.
    """

    def __init__(self, source_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a webcam source.

        Args:
            source_id: Unique identifier for this source.
            config: Configuration dictionary containing:
                - device_id: Camera device ID (default: 0).
                - fps: FPS to process frames (default: 30).
                - resolution: Resolution to set for the camera (default: (640, 480)).
                - api_preference: OpenCV camera API preference (default: AUTO).
        """
        super().__init__(source_id, config)
        self.source_type = SourceType.WEBCAM

        # Webcam specific configs
        self.device_id = self.config.get("device_id", 0)
        self.api_preference = self.config.get("api_preference", cv2.CAP_ANY)
        self.cap = None

        # Map string API preference to cv2 constant if provided as string
        if isinstance(self.api_preference, str):
            api_map = {
                "ANY": cv2.CAP_ANY,
                "VFW": cv2.CAP_VFW,
                "V4L": cv2.CAP_V4L,
                "V4L2": cv2.CAP_V4L2,
                "FIREWIRE": cv2.CAP_FIREWIRE,
                "DSHOW": cv2.CAP_DSHOW,
                "MSMF": cv2.CAP_MSMF,
                "AUTO": cv2.CAP_ANY,
            }
            self.api_preference = api_map.get(self.api_preference.upper(), cv2.CAP_ANY)

    def open(self) -> bool:
        """
        Open the webcam.

        Returns:
            True if webcam opened successfully, False otherwise.
        """
        try:
            # Open the webcam with specified API
            if isinstance(self.device_id, str) and self.device_id.isdigit():
                self.cap = cv2.VideoCapture(int(self.device_id), self.api_preference)
            else:
                self.cap = cv2.VideoCapture(self.device_id, self.api_preference)

            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open webcam (ID: {self.device_id})")

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Try to set FPS if supported
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Get actual camera properties
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            # Read a test frame to confirm working connection
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError(
                    f"Could not read initial frame from webcam (ID: {self.device_id})"
                )

            self.is_opened = True
            return True

        except Exception as e:
            print(f"Error opening webcam source {self.source_id}: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_opened = False
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the webcam.

        Returns:
            Tuple of (success, frame).
        """
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()

        if not ret:
            return False, None

        # Resize frame if the actual resolution differs from the requested one
        if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
            frame = cv2.resize(frame, self.resolution)

        return True, frame

    def close(self) -> None:
        """Close the webcam."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the webcam.

        Returns:
            Dictionary with information about the webcam.
        """
        info = super().get_info()
        info.update(
            {
                "device_id": self.device_id,
                "api_preference": self.api_preference,
                "actual_resolution": (
                    getattr(self, "actual_width", None),
                    getattr(self, "actual_height", None),
                ),
                "actual_fps": getattr(self, "actual_fps", None),
            }
        )
        return info

    @staticmethod
    def list_available_cameras() -> List[Dict[str, Any]]:
        """
        List all available camera devices.

        Returns:
            List of dictionaries with camera information.
        """
        cameras = []

        # Check the first 10 indexes
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                cameras.append(
                    {"device_id": i, "resolution": (width, height), "fps": fps}
                )

                cap.release()

        return cameras
