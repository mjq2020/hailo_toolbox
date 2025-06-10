"""
MIPI camera source implementation.

This module handles MIPI CSI-2 cameras commonly used with embedded systems like 
Raspberry Pi, Jetson Nano, and other SoCs.
"""

import os
import cv2
import time
import numpy as np
import importlib.util
from typing import Dict, Any, Optional, Tuple, List
from .base import BaseSource, SourceType


class MIPICameraSource(BaseSource):
    """
    Video source for MIPI cameras on embedded platforms.
    
    Supports both GStreamer-based pipelines and platform-specific camera APIs.
    """
    
    def __init__(self, source_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a MIPI camera source.
        
        Args:
            source_id: Unique identifier for this source.
            config: Configuration dictionary containing:
                - pipeline_type: Type of pipeline ("gstreamer", "jetson", "libcamera", "v4l2", etc.).
                - sensor_id: Camera sensor ID (default: 0).
                - fps: FPS to process frames (default: 30).
                - resolution: Resolution for the camera (default: (640, 480)).
                - format: Pixel format (default: "BGR").
                - flip_method: Flip method for GStreamer (default: 0).
                - sensor_mode: Sensor mode for Jetson cameras (default: 0).
                - custom_pipeline: Custom GStreamer pipeline string.
        """
        super().__init__(source_id, config)
        self.source_type = SourceType.MIPI_CAMERA
        
        # MIPI camera specific configs
        self.pipeline_type = self.config.get("pipeline_type", "gstreamer")
        self.sensor_id = self.config.get("sensor_id", 0)
        self.flip_method = self.config.get("flip_method", 0)
        self.sensor_mode = self.config.get("sensor_mode", 0)
        self.custom_pipeline = self.config.get("custom_pipeline")
        
        # Hardware specific objects
        self.cap = None
        self.camera = None  # For platform-specific APIs
        
        # Check available backends
        self._check_available_backends()
        
    def _check_available_backends(self):
        """Check which camera backends are available on the system."""
        self.available_backends = {}
        
        # Check for GStreamer
        try:
            if cv2.getBuildInformation().find("GStreamer") != -1:
                self.available_backends["gstreamer"] = True
        except:
            self.available_backends["gstreamer"] = False
            
        # Check for Jetson-specific camera API
        try:
            spec = importlib.util.find_spec("jetson.utils")
            self.available_backends["jetson"] = spec is not None
        except:
            self.available_backends["jetson"] = False
            
        # Check for libcamera-py
        try:
            spec = importlib.util.find_spec("libcamera")
            self.available_backends["libcamera"] = spec is not None
        except:
            self.available_backends["libcamera"] = False
            
        # Check for Raspberry Pi specific API (picamera2)
        try:
            spec = importlib.util.find_spec("picamera2")
            self.available_backends["picamera2"] = spec is not None
        except:
            self.available_backends["picamera2"] = False
            
    def _create_gstreamer_pipeline(self) -> str:
        """
        Create a GStreamer pipeline string for the MIPI camera.
        
        Returns:
            GStreamer pipeline string.
        """
        if self.custom_pipeline:
            return self.custom_pipeline
            
        width, height = self.resolution
        
        # Default pipeline for NVIDIA Jetson platforms
        if os.path.exists("/dev/nvhost-ctrl"):  # Simple check for Jetson
            return (
                f"nvarguscamerasrc sensor-id={self.sensor_id} sensor-mode={self.sensor_mode} ! "
                f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={self.fps}/1 ! "
                f"nvvidconv flip-method={self.flip_method} ! "
                f"video/x-raw, width={width}, height={height}, format=BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink"
            )
        
        # Default pipeline for Raspberry Pi with v4l2
        elif os.path.exists("/opt/vc/lib/libmmal.so"):  # Simple check for Raspberry Pi
            return (
                f"v4l2src device=/dev/video{self.sensor_id} ! "
                f"video/x-raw, width={width}, height={height}, framerate={self.fps}/1 ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink"
            )
            
        # Generic v4l2 pipeline
        else:
            return (
                f"v4l2src device=/dev/video{self.sensor_id} ! "
                f"video/x-raw, width={width}, height={height}, framerate={self.fps}/1 ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink"
            )
            
    def _open_gstreamer(self) -> bool:
        """
        Open the MIPI camera using GStreamer.
        
        Returns:
            True if opened successfully, False otherwise.
        """
        try:
            pipeline = self._create_gstreamer_pipeline()
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open MIPI camera with GStreamer pipeline: {pipeline}")
                
            # Read a test frame
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Could not read initial frame from MIPI camera")
                
            return True
        except Exception as e:
            print(f"Error opening MIPI camera with GStreamer: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False
            
    def _open_jetson(self) -> bool:
        """
        Open the MIPI camera using Jetson-specific APIs.
        
        Returns:
            True if opened successfully, False otherwise.
        """
        try:
            from jetson.utils import videoSource, cudaToNumpy
            from jetson.utils import videoOutput
            
            width, height = self.resolution
            pipeline = f"csi://{self.sensor_id}?width={width}&height={height}&framerate={self.fps}"
            
            self.camera = videoSource(pipeline)
            self._cuda_frame = None  # Will store CUDA frame
            
            return True
        except Exception as e:
            print(f"Error opening MIPI camera with Jetson API: {e}")
            if hasattr(self, "camera") and self.camera is not None:
                del self.camera
                self.camera = None
            return False
            
    def _open_picamera2(self) -> bool:
        """
        Open the MIPI camera using Raspberry Pi's picamera2.
        
        Returns:
            True if opened successfully, False otherwise.
        """
        try:
            from picamera2 import Picamera2
            
            self.camera = Picamera2(self.sensor_id)
            width, height = self.resolution
            
            config = self.camera.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            
            # Give camera time to start
            time.sleep(1.0)
            
            return True
        except Exception as e:
            print(f"Error opening MIPI camera with picamera2: {e}")
            if hasattr(self, "camera") and self.camera is not None:
                self.camera.close()
                self.camera = None
            return False
            
    def _open_libcamera(self) -> bool:
        """
        Open the MIPI camera using libcamera.
        
        Returns:
            True if opened successfully, False otherwise.
        """
        try:
            # This is a simplified example, actual implementation would be more complex
            import libcamera
            
            width, height = self.resolution
            
            # Configure libcamera (simplified)
            self.camera = libcamera.Camera()
            self.camera.configure_stream(width, height, self.fps)
            self.camera.start()
            
            return True
        except Exception as e:
            print(f"Error opening MIPI camera with libcamera: {e}")
            if hasattr(self, "camera") and self.camera is not None:
                self.camera.stop()
                self.camera = None
            return False
    
    def open(self) -> bool:
        """
        Open the MIPI camera.
        
        Returns:
            True if camera opened successfully, False otherwise.
        """
        success = False
        
        # Try with the specified pipeline type first
        if self.pipeline_type == "gstreamer" and self.available_backends.get("gstreamer", False):
            success = self._open_gstreamer()
        elif self.pipeline_type == "jetson" and self.available_backends.get("jetson", False):
            success = self._open_jetson()
        elif self.pipeline_type == "picamera2" and self.available_backends.get("picamera2", False):
            success = self._open_picamera2()
        elif self.pipeline_type == "libcamera" and self.available_backends.get("libcamera", False):
            success = self._open_libcamera()
            
        # If the specified backend failed, try alternatives
        if not success:
            # Try GStreamer as a fallback
            if self.available_backends.get("gstreamer", False) and self.pipeline_type != "gstreamer":
                print(f"Falling back to GStreamer for MIPI camera {self.source_id}")
                self.pipeline_type = "gstreamer"
                success = self._open_gstreamer()
                
            # Try Jetson-specific API as a fallback
            elif self.available_backends.get("jetson", False) and self.pipeline_type != "jetson":
                print(f"Falling back to Jetson API for MIPI camera {self.source_id}")
                self.pipeline_type = "jetson"
                success = self._open_jetson()
                
            # Try picamera2 as a fallback
            elif self.available_backends.get("picamera2", False) and self.pipeline_type != "picamera2":
                print(f"Falling back to picamera2 for MIPI camera {self.source_id}")
                self.pipeline_type = "picamera2"
                success = self._open_picamera2()
                
        self.is_opened = success
        return success
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the MIPI camera.
        
        Returns:
            Tuple of (success, frame).
        """
        if not self.is_opened:
            return False, None
            
        try:
            # GStreamer pipeline using OpenCV
            if self.pipeline_type == "gstreamer" and self.cap is not None:
                ret, frame = self.cap.read()
                if not ret:
                    return False, None
                    
            # Jetson-specific API
            elif self.pipeline_type == "jetson" and self.camera is not None:
                from jetson.utils import cudaToNumpy
                
                self._cuda_frame = self.camera.Capture()
                frame = cudaToNumpy(self._cuda_frame)
                
                # Convert from RGBA to BGR if needed
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    
                ret = True
                
            # picamera2 API
            elif self.pipeline_type == "picamera2" and self.camera is not None:
                frame = self.camera.capture_array()
                
                # Convert from RGB to BGR if needed
                if self.format.upper() == "BGR" and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                ret = True
                
            # libcamera API
            elif self.pipeline_type == "libcamera" and self.camera is not None:
                # Simplified example
                frame = self.camera.capture_image()
                
                # Convert to NumPy array and BGR format if needed
                frame = np.array(frame)
                if self.format.upper() == "BGR" and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                ret = True
                
            else:
                return False, None
                
            # Resize if necessary
            if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
                frame = cv2.resize(frame, self.resolution)
                
            return ret, frame
            
        except Exception as e:
            print(f"Error reading frame from MIPI camera {self.source_id}: {e}")
            return False, None
    
    def close(self) -> None:
        """Close the MIPI camera."""
        try:
            # GStreamer pipeline
            if self.pipeline_type == "gstreamer" and self.cap is not None:
                self.cap.release()
                
            # Jetson-specific API
            elif self.pipeline_type == "jetson" and self.camera is not None:
                del self.camera
                
            # picamera2 API
            elif self.pipeline_type == "picamera2" and self.camera is not None:
                self.camera.close()
                
            # libcamera API
            elif self.pipeline_type == "libcamera" and self.camera is not None:
                self.camera.stop()
                
        except Exception as e:
            print(f"Error closing MIPI camera {self.source_id}: {e}")
            
        finally:
            self.cap = None
            self.camera = None
            self.is_opened = False
        
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the MIPI camera.
        
        Returns:
            Dictionary with information about the MIPI camera.
        """
        info = super().get_info()
        info.update({
            "pipeline_type": self.pipeline_type,
            "sensor_id": self.sensor_id,
            "flip_method": self.flip_method,
            "sensor_mode": self.sensor_mode,
            "available_backends": self.available_backends
        })
        return info 