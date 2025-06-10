"""
File-based video source implementation.
"""

import os
import cv2
import glob
import requests
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from urllib.parse import urlparse
from .base import BaseSource, SourceType


class ImageSource(BaseSource):
    """
    Image source that reads from a local image file, folder, or URL.
    
    Supports single images, image folders, and network image URLs.
    """
    
    def __init__(self, source_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize an image source.
        
        Args:
            source_id: Unique identifier for this source.
            config: Configuration dictionary containing:
                - source_path: Path to image file, folder, or URL.
                - supported_formats: List of supported image formats (default: common formats).
                - loop: Whether to loop through images in folder (default: True).
                - sort_files: Whether to sort files in folder (default: True).
                - timeout: Timeout for URL requests in seconds (default: 10).
        """
        super().__init__(source_id, config)
        self.source_type = SourceType.IMAGE
        
        # Image specific configs
        self.source_path = self.config.get("source_path")
        if not self.source_path:
            raise ValueError("source_path must be provided for ImageSource")
            
        self.supported_formats = self.config.get("supported_formats", [
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"
        ])
        self.loop = self.config.get("loop", True)
        self.sort_files = self.config.get("sort_files", True)
        self.timeout = self.config.get("timeout", 10)
        
        # Internal state
        self.image_files = []
        self.current_index = 0
        self.is_url = self._is_url(self.source_path)
        self.is_folder = False
        self.current_image = None
        
    def _is_url(self, path: str) -> bool:
        """Check if the path is a URL."""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False
            
    def _load_image_from_url(self, url: str) -> Optional[np.ndarray]:
        """
        Load image from URL.
        
        Args:
            url: Image URL.
            
        Returns:
            Image as numpy array or None if failed.
        """
        try:
            response = requests.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            # Convert response content to numpy array
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError(f"Failed to decode image from URL: {url}")
                
            return image
            
        except Exception as e:
            print(f"Error loading image from URL {url}: {e}")
            return None
            
    def _load_image_from_file(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load image from local file.
        
        Args:
            file_path: Path to image file.
            
        Returns:
            Image as numpy array or None if failed.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file not found: {file_path}")
                
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to load image: {file_path}")
                
            return image
            
        except Exception as e:
            print(f"Error loading image from file {file_path}: {e}")
            return None
            
    def _scan_folder(self, folder_path: str) -> List[str]:
        """
        Scan folder for image files.
        
        Args:
            folder_path: Path to folder.
            
        Returns:
            List of image file paths.
        """
        image_files = []
        
        for ext in self.supported_formats:
            # Case insensitive search
            pattern = os.path.join(folder_path, f"*{ext}")
            image_files.extend(glob.glob(pattern))
            pattern = os.path.join(folder_path, f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern))
            
        # Remove duplicates and sort if requested
        image_files = list(set(image_files))
        if self.sort_files:
            image_files.sort()
            
        return image_files
        
    def open(self) -> bool:
        """
        Open the image source.
        
        Returns:
            True if source opened successfully, False otherwise.
        """
        try:
            if self.is_url:
                # Single image URL
                self.image_files = [self.source_path]
                self.is_folder = False
            else:
                # Local path
                path = Path(self.source_path)
                
                if path.is_file():
                    # Single image file
                    if path.suffix.lower() not in self.supported_formats:
                        raise ValueError(f"Unsupported image format: {path.suffix}")
                    self.image_files = [str(path)]
                    self.is_folder = False
                    
                elif path.is_dir():
                    # Image folder
                    self.image_files = self._scan_folder(str(path))
                    if not self.image_files:
                        raise ValueError(f"No supported image files found in folder: {path}")
                    self.is_folder = True
                    
                else:
                    raise FileNotFoundError(f"Path does not exist: {path}")
                    
            self.current_index = 0
            self.is_opened = True
            return True
            
        except Exception as e:
            print(f"Error opening image source {self.source_id}: {e}")
            self.is_opened = False
            return False
            
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next image.
        
        Returns:
            Tuple of (success, image).
        """
        if not self.is_opened or not self.image_files:
            return False, None
            
        # Check if we've reached the end
        if self.current_index >= len(self.image_files):
            if self.loop and self.is_folder:
                self.current_index = 0  # Reset to beginning
            else:
                return False, None  # End of images
                
        # Load current image
        current_path = self.image_files[self.current_index]
        
        if self.is_url:
            image = self._load_image_from_url(current_path)
        else:
            image = self._load_image_from_file(current_path)
            
        if image is None:
            # Skip this image and try next
            self.current_index += 1
            return self.read()
            
        # Resize if needed
        if image.shape[1] != self.resolution[0] or image.shape[0] != self.resolution[1]:
            image = cv2.resize(image, self.resolution)
            
        self.current_image = image
        self.current_index += 1
        
        return True, image
        
    def close(self) -> None:
        """Close the image source."""
        self.current_image = None
        self.is_opened = False
        
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the image source.
        
        Returns:
            Dictionary with information about the image source.
        """
        info = super().get_info()
        info.update({
            "source_path": self.source_path,
            "is_url": self.is_url,
            "is_folder": self.is_folder,
            "total_images": len(self.image_files),
            "current_index": self.current_index,
            "supported_formats": self.supported_formats,
            "loop": self.loop
        })
        return info
        
    def seek(self, index: int) -> bool:
        """
        Seek to a specific image index (only for folders).
        
        Args:
            index: Image index to seek to.
            
        Returns:
            True if seek was successful, False otherwise.
        """
        if not self.is_opened or not self.is_folder:
            return False
            
        if 0 <= index < len(self.image_files):
            self.current_index = index
            return True
            
        return False


class VideoSource(BaseSource):
    """
    Video source that reads from a local video file or network video stream.
    """

    def __init__(self, source_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a file-based video source.

        Args:
            source_id: Unique identifier for this source.
            config: Configuration dictionary containing:
                - file_path: Path to the video file or network URL.
                - loop: Whether to loop the video (default: False).
                - start_frame: Starting frame index (default: 0).
                - fps: FPS to use (default: video's original FPS).
                - resolution: Resolution to resize frames to (default: original).
                - timeout: Timeout for network streams in seconds (default: 10).
        """
        super().__init__(source_id, config)
        self.source_type = SourceType.FILE

        # File specific configs
        self.file_path = self.config.get("file_path")
        if not self.file_path:
            raise ValueError("file_path must be provided for VideoSource")

        self.loop = self.config.get("loop", False)
        self.start_frame = self.config.get("start_frame", 0)
        self.timeout = self.config.get("timeout", 10)
        self.current_frame_idx = self.start_frame
        self.cap = None

        # Check if it's a network stream
        self.is_network_stream = self._is_network_stream(self.file_path)
        
        if not self.is_network_stream:
            # Extract file info for local files
            self.file_name = os.path.basename(self.file_path)
            self.file_extension = os.path.splitext(self.file_name)[1].lower()

            # Check supported formats for local files
            self.supported_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"]
            if self.file_extension not in self.supported_extensions:
                raise ValueError(
                    f"Unsupported file format: {self.file_extension}. Supported formats: {self.supported_extensions}"
                )
        else:
            self.file_name = self.file_path
            self.file_extension = "network_stream"

    def _is_network_stream(self, path: str) -> bool:
        """Check if the path is a network stream URL."""
        network_protocols = ["http://", "https://", "rtsp://", "rtmp://", "udp://", "tcp://"]
        return any(path.startswith(protocol) for protocol in network_protocols)

    def open(self) -> bool:
        """
        Open the video file or network stream.

        Returns:
            True if file opened successfully, False otherwise.
        """
        try:
            if not self.is_network_stream and not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")

            # Set up capture with appropriate backend
            if self.is_network_stream:
                self.cap = cv2.VideoCapture(self.file_path, cv2.CAP_FFMPEG)
                # Set timeout for network streams
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout * 1000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout * 1000)
            else:
                self.cap = cv2.VideoCapture(self.file_path)

            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {self.file_path}")

            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Use original FPS if not specified
            if "fps" not in self.config:
                self.fps = self.original_fps if self.original_fps > 0 else 30

            # Use original resolution if not specified
            if "resolution" not in self.config:
                self.resolution = (self.original_width, self.original_height)

            # Set starting frame for local files
            if not self.is_network_stream and self.start_frame > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                self.current_frame_idx = self.start_frame

            self.is_opened = True
            return True

        except Exception as e:
            print(f"Error opening video source {self.source_id}: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_opened = False
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the video file or stream.

        Returns:
            Tuple of (success, frame).
        """
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        
        if not self.is_network_stream:
            self.current_frame_idx += 1

        # Handle end of video
        if not ret:
            if self.loop and not self.is_network_stream and self.total_frames > 0:
                # Reset to beginning for looping (only for local files)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                self.current_frame_idx = self.start_frame
                ret, frame = self.cap.read()
                if not ret:
                    return False, None
            else:
                return False, None

        # Resize frame if needed
        if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
            frame = cv2.resize(frame, self.resolution)

        return True, frame

    def close(self) -> None:
        """Close the video file or stream."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the video source.

        Returns:
            Dictionary with information about the video source.
        """
        info = super().get_info()
        info.update(
            {
                "file_path": self.file_path,
                "file_name": getattr(self, "file_name", ""),
                "file_extension": getattr(self, "file_extension", ""),
                "is_network_stream": self.is_network_stream,
                "loop": self.loop,
                "current_frame": self.current_frame_idx,
                "total_frames": getattr(self, "total_frames", None),
                "original_fps": getattr(self, "original_fps", None),
                "original_resolution": (
                    getattr(self, "original_width", None),
                    getattr(self, "original_height", None),
                ),
            }
        )
        return info

    def seek(self, frame_idx: int) -> bool:
        """
        Seek to a specific frame in the video (only for local files).

        Args:
            frame_idx: Frame index to seek to.

        Returns:
            True if seek was successful, False otherwise.
        """
        if not self.is_opened or self.cap is None or self.is_network_stream:
            return False

        if frame_idx < 0 or (self.total_frames > 0 and frame_idx >= self.total_frames):
            return False

        success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        if success:
            self.current_frame_idx = frame_idx

        return success


# Backward compatibility
FileSource = VideoSource
