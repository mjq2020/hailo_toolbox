"""
Base class for all inference engines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable, Union, TypeVar, Generic
import numpy as np
import time


# Type for the inference callback function
InferenceCallback = Callable[['InferenceResult'], Any]


@dataclass
class InferenceResult:
    """
    Class to hold the results of an inference operation.
    """
    # Required fields
    success: bool
    model_name: str
    raw_outputs: Dict[str, np.ndarray]
    
    # Optional fields
    input_data: Optional[Dict[str, np.ndarray]] = None
    processed_outputs: Any = None
    inference_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    frame_id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def total_time_ms(self) -> float:
        """
        Calculate the total processing time.
        
        Returns:
            Total time in milliseconds.
        """
        return self.preprocessing_time_ms + self.inference_time_ms + self.postprocessing_time_ms


class BaseInferenceEngine(ABC):
    """
    Abstract base class for all inference engines.
    
    This class defines the common interface that all inference engines must implement.
    """
    
    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the model file.
            config: Optional configuration dictionary.
        """
        self.model_path = model_path
        self.config = config or {}
        self.is_loaded = False
        self.model_name = self.config.get("model_name", "unknown")
        
        # Get preprocessing and postprocessing config
        self.preprocess_config = self.config.get("preprocess", {})
        self.postprocess_config = self.config.get("postprocess", {})
        
        # Callback function to be called on inference results
        self.callback: Optional[InferenceCallback] = self.config.get("callback")
        
    @abstractmethod
    def load(self) -> bool:
        """
        Load the model.
        
        Returns:
            True if the model was loaded successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free resources."""
        pass
    
    @abstractmethod
    def infer(self, input_data: Dict[str, np.ndarray]) -> InferenceResult:
        """
        Run inference on the input data.
        
        Args:
            input_data: Dictionary mapping input names to numpy arrays.
            
        Returns:
            InferenceResult containing the results.
        """
        pass
    
    def preprocess(self, data: Any) -> Dict[str, np.ndarray]:
        """
        Preprocess data before inference.
        
        Args:
            data: Data to preprocess (e.g., image, text, etc.).
            
        Returns:
            Dictionary mapping input names to preprocessed numpy arrays.
        """
        # Default implementation does basic image preprocessing
        if isinstance(data, np.ndarray) and len(data.shape) == 3:
            # Basic image preprocessing
            input_name = self.config.get("input_name", "input")
            input_shape = self.config.get("input_shape")
            
            # Start timing preprocessing
            start_time = time.time()
            
            # Resize if input shape is provided
            if input_shape and len(input_shape) >= 3:
                h, w = input_shape[1:3] if len(input_shape) == 4 else input_shape[0:2]
                if data.shape[0] != h or data.shape[1] != w:
                    import cv2
                    data = cv2.resize(data, (w, h))
            
            # Normalize if specified
            normalize = self.preprocess_config.get("normalize", False)
            if normalize:
                scale = self.preprocess_config.get("scale", 1.0/255.0)
                mean = self.preprocess_config.get("mean", [0.485, 0.456, 0.406])
                std = self.preprocess_config.get("std", [0.229, 0.224, 0.225])
                
                data = data.astype(np.float32) * scale
                if len(mean) == 3 and data.shape[2] == 3:
                    data = (data - np.array(mean)) / np.array(std)
            
            # Change layout if needed (HWC -> NCHW)
            layout = self.preprocess_config.get("layout", "NCHW")
            if layout == "NCHW" and len(data.shape) == 3:
                data = np.transpose(data, (2, 0, 1))  # HWC -> CHW
                data = np.expand_dims(data, axis=0)  # CHW -> NCHW
                
            # Calculate preprocessing time
            preprocessing_time_ms = (time.time() - start_time) * 1000
            
            result = {input_name: data}
            # Store preprocessing time for later use in infer()
            result["__preprocessing_time_ms"] = preprocessing_time_ms
            
            return result
        
        # For other types of data, return as is
        return {"input": data}
    
    def postprocess(self, result: InferenceResult) -> InferenceResult:
        """
        Postprocess the inference result.
        
        Args:
            result: Inference result to postprocess.
            
        Returns:
            Postprocessed InferenceResult.
        """
        # Default implementation simply returns the raw outputs
        result.processed_outputs = result.raw_outputs
        return result
    
    def __call__(self, data: Any) -> InferenceResult:
        """
        Convenience method to preprocess, infer, and postprocess in one call.
        
        Args:
            data: Input data (e.g., image).
            
        Returns:
            Processed InferenceResult.
        """
        # Preprocess the data
        input_data = self.preprocess(data)
        
        # Extract preprocessing time if available
        preprocessing_time_ms = input_data.pop("__preprocessing_time_ms", 0.0)
        
        # Run inference
        result = self.infer(input_data)
        
        # Add preprocessing time
        result.preprocessing_time_ms = preprocessing_time_ms
        
        # Postprocess the result
        start_time = time.time()
        result = self.postprocess(result)
        result.postprocessing_time_ms = (time.time() - start_time) * 1000
        
        # Call the callback if provided
        if self.callback is not None:
            self.callback(result)
            
        return result
    
    def is_model_loaded(self) -> bool:
        """
        Check if the model is loaded.
        
        Returns:
            True if the model is loaded, False otherwise.
        """
        return self.is_loaded
    
    def set_callback(self, callback: InferenceCallback) -> None:
        """
        Set a callback function to be called on inference results.
        
        Args:
            callback: Callback function taking an InferenceResult.
        """
        self.callback = callback
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "is_loaded": self.is_loaded
        }
        
    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload() 