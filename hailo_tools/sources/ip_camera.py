"""
IP camera video source implementation.
"""

import cv2
import time
import threading
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Deque
from collections import deque
from .base import BaseSource, SourceType


class IPCameraSource(BaseSource):
    """
    Video source for IP cameras (RTSP, HTTP streams, etc.).
    
    Includes buffering to handle network issues and maintain a smooth stream.
    """
    
    def __init__(self, source_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize an IP camera source.
        
        Args:
            source_id: Unique identifier for this source.
            config: Configuration dictionary containing:
                - url: RTSP or HTTP URL for the camera stream.
                - username: Optional username for authentication.
                - password: Optional password for authentication.
                - buffer_size: Size of the frame buffer (default: 30).
                - reconnect_attempts: Number of reconnection attempts (default: 3).
                - reconnect_delay: Delay between reconnection attempts in seconds (default: 5).
                - timeout: Timeout for connection in seconds (default: 10).
                - fps: FPS to process frames (default: 30).
                - resolution: Resolution to resize frames to (default: (640, 480)).
        """
        super().__init__(source_id, config)
        self.source_type = SourceType.IP_CAMERA
        
        # IP camera specific configs
        self.url = self.config.get("url")
        if not self.url:
            raise ValueError("url must be provided for IPCameraSource")
            
        self.username = self.config.get("username")
        self.password = self.config.get("password")
        
        # Format URL with authentication if provided
        if self.username and self.password and "://" in self.url:
            protocol, rest = self.url.split("://", 1)
            self.url = f"{protocol}://{self.username}:{self.password}@{rest}"
            
        # Buffering and reconnection settings
        self.buffer_size = self.config.get("buffer_size", 30)
        self.reconnect_attempts = self.config.get("reconnect_attempts", 3)
        self.reconnect_delay = self.config.get("reconnect_delay", 5)
        self.timeout = self.config.get("timeout", 10)
        
        # Stream handling
        self.cap = None
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.buffer_thread = None
        self.stop_event = threading.Event()
        self.buffer_lock = threading.Lock()
        
    def open(self) -> bool:
        """
        Open the IP camera stream.
        
        Returns:
            True if stream opened successfully, False otherwise.
        """
        try:
            # Set OpenCV options for streaming
            stream_options = {
                cv2.CAP_PROP_BUFFERSIZE: self.buffer_size,
                cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'MJPG'),  # Try MJPEG format
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC: self.timeout * 1000,  # Convert to milliseconds
                cv2.CAP_PROP_READ_TIMEOUT_MSEC: self.timeout * 1000   # Convert to milliseconds
            }
            
            # Open the camera stream
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)  # Explicitly use FFMPEG backend
            
            # Set stream options
            for option, value in stream_options.items():
                self.cap.set(option, value)
                
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open IP camera stream: {self.url}")
                
            # Read a test frame to confirm working connection
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError(f"Could not read initial frame from IP camera: {self.url}")
                
            # Get stream properties
            self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Use original resolution if not specified
            if "resolution" not in self.config:
                self.resolution = (self.original_width, self.original_height)
                
            # Start buffer thread
            self.stop_event.clear()
            self.buffer_thread = threading.Thread(target=self._buffer_frames, daemon=True)
            self.buffer_thread.start()
            
            self.is_opened = True
            return True
            
        except Exception as e:
            print(f"Error opening IP camera source {self.source_id}: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_opened = False
            return False
            
    def _buffer_frames(self) -> None:
        """
        Background thread to continuously buffer frames from the camera.
        """
        reconnect_count = 0
        
        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                # Try to reconnect
                if reconnect_count < self.reconnect_attempts:
                    print(f"Reconnecting to IP camera {self.source_id} (attempt {reconnect_count + 1}/{self.reconnect_attempts})...")
                    reconnect_count += 1
                    
                    if self.cap is not None:
                        self.cap.release()
                    
                    self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                    if not self.cap.isOpened():
                        print(f"Failed to reconnect to IP camera {self.source_id}")
                        time.sleep(self.reconnect_delay)
                        continue
                        
                    reconnect_count = 0  # Reset counter on successful reconnection
                else:
                    print(f"Failed to reconnect to IP camera {self.source_id} after {self.reconnect_attempts} attempts")
                    self.stop_event.set()
                    break
            
            # Read a new frame
            ret, frame = self.cap.read()
            
            if not ret:
                time.sleep(0.1)  # Short sleep to prevent CPU hogging
                continue
                
            # Resize frame if needed
            if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
                frame = cv2.resize(frame, self.resolution)
                
            # Add to buffer with thread safety
            with self.buffer_lock:
                self.frame_buffer.append((ret, frame))
                
            # Control the buffering rate to match the desired FPS
            time.sleep(1.0 / (self.fps * 2))  # Buffer at 2x the playback rate for smoothness
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the IP camera.
        
        Returns:
            Tuple of (success, frame).
        """
        if not self.is_opened:
            return False, None
            
        # Get frame from buffer if available
        with self.buffer_lock:
            if len(self.frame_buffer) > 0:
                return self.frame_buffer.popleft()
            
        # If buffer is empty but camera is still open, wait briefly for new frames
        if self.cap is not None and self.cap.isOpened():
            time.sleep(0.1)  # Short sleep
            
            # Try again
            with self.buffer_lock:
                if len(self.frame_buffer) > 0:
                    return self.frame_buffer.popleft()
                    
        # If still no frames, read directly from camera as fallback
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and (frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]):
                frame = cv2.resize(frame, self.resolution)
            return ret, frame
                
        return False, None
    
    def close(self) -> None:
        """Close the IP camera stream."""
        # Stop the buffer thread
        self.stop_event.set()
        if self.buffer_thread is not None and self.buffer_thread.is_alive():
            self.buffer_thread.join(timeout=2.0)  # Wait for thread to finish with timeout
            
        # Clear the buffer
        with self.buffer_lock:
            self.frame_buffer.clear()
            
        # Release the capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        self.is_opened = False
        
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the IP camera.
        
        Returns:
            Dictionary with information about the IP camera.
        """
        info = super().get_info()
        
        # Remove sensitive information
        safe_url = self.url
        if self.username and self.password and self.username in safe_url:
            protocol, rest = safe_url.split("://", 1)
            auth_rest = rest.split("@", 1)
            if len(auth_rest) > 1:
                safe_url = f"{protocol}://****:****@{auth_rest[1]}"
        
        info.update({
            "url": safe_url,
            "has_credentials": bool(self.username and self.password),
            "buffer_size": self.buffer_size,
            "reconnect_attempts": self.reconnect_attempts,
            "buffer_length": len(self.frame_buffer) if hasattr(self, "frame_buffer") else 0,
            "original_resolution": (
                getattr(self, "original_width", None),
                getattr(self, "original_height", None)
            ),
            "original_fps": getattr(self, "original_fps", None)
        })
        return info 