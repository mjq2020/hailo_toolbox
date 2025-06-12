from typing import Any, List, Optional, AnyStr, Union, Callable, Dict, Type
from enum import Enum

from hailo_toolbox.inference.hailo_engine import HailoInference
from hailo_toolbox.inference.onnx_engine import ONNXInference
from hailo_toolbox.utils.config import Config
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CallbackType(Enum):
    """Enumeration of callback types supported by the registry"""

    PRE_PROCESSOR = "pre_processor"
    POST_PROCESSOR = "post_processor"
    VISUALIZER = "visualizer"
    SOURCE = "source"
    COLLAT_INFER = "collat_infer"


def empty_callback(*args, **kwargs) -> None:
    """Default empty callback function that does nothing

    Returns:
        None
    """
    pass


class CallbackRegistry:
    """
    A universal registry for managing callbacks/functions/classes by name and type.

    This registry provides a centralized way to register and retrieve different types
    of callbacks, functions, or classes based on their name and category type.

    Features:
    - Type-safe registration and retrieval
    - Support for decorator-style registration
    - Extensible callback types via Enum
    - Automatic fallback to empty callback for missing entries
    - Comprehensive error handling and validation

    Example:
        # Create registry instance
        registry = CallbackRegistry()

        # Register using decorator
        @registry.register("my_preprocessor", CallbackType.PRE_PROCESSOR)
        def my_preprocess_func(data):
            return processed_data

        # Register directly
        registry.register_callback("my_postprocessor", CallbackType.POST_PROCESSOR, my_postprocess_func)

        # Retrieve registered callback
        preprocessor = registry.get_callback("my_preprocessor", CallbackType.PRE_PROCESSOR)
    """

    def __init__(self):
        """Initialize the callback registry with empty dictionaries for each type"""
        # Main storage: Dict[CallbackType, Dict[str, Union[Callable, Type]]]
        self._callbacks: Dict[CallbackType, Dict[str, Union[Callable, Type]]] = {
            callback_type: {} for callback_type in CallbackType
        }

        # Legacy compatibility - maintain separate dictionaries
        self.engines: Dict[str, Any] = {}
        self.PreProcessor: Dict[str, Callable] = {}
        self.PostProcessor: Dict[str, Callable] = {}
        self.Visualizer: Dict[str, Callable] = {}
        self.Source: Dict[str, Callable] = {}
        self.CollatInfer: Dict[str, Callable] = {}

    def register(self, name: str, callback_type: CallbackType) -> Callable:
        """
        Decorator for registering callbacks with specified name and type.

        Args:
            name: Unique identifier for the callback
            callback_type: Type of callback from CallbackType enum

        Returns:
            Decorator function

        Example:
            @registry.register("yolov8_preprocess", CallbackType.PRE_PROCESSOR)
            def preprocess_func(data):
                return processed_data
        """

        def decorator(func: Union[Callable, Type]) -> Union[Callable, Type]:
            self.register_callback(name, callback_type, func)
            return func

        return decorator

    def register_callback(
        self, name: str, callback_type: CallbackType, callback: Union[Callable, Type]
    ) -> None:
        """
        Register a callback/function/class with the specified name and type.

        Args:
            name: Unique identifier for the callback
            callback_type: Type of callback from CallbackType enum
            callback: Function, method, or class to register

        Raises:
            ValueError: If callback_type is not a valid CallbackType
            TypeError: If callback is not callable or a class
            KeyError: If name already exists for the given type (overwrites with warning)
        """
        if not isinstance(callback_type, CallbackType):
            raise ValueError(
                f"Invalid callback type: {callback_type}. Must be a CallbackType enum."
            )

        if not (callable(callback) or isinstance(callback, type)):
            raise TypeError(
                f"Callback must be callable or a class, got {type(callback)}"
            )

        # Check for existing registration
        if name in self._callbacks[callback_type]:
            logger.warning(
                f"Overwriting existing callback '{name}' for type '{callback_type.value}'"
            )

        # Register in main storage
        self._callbacks[callback_type][name] = callback

        # Maintain legacy compatibility
        self._update_legacy_storage(name, callback_type, callback)

        logger.debug(f"Registered callback '{name}' for type '{callback_type.value}'")

    def _update_legacy_storage(
        self, name: str, callback_type: CallbackType, callback: Union[Callable, Type]
    ) -> None:
        """Update legacy storage dictionaries for backward compatibility"""
        if callback_type == CallbackType.PRE_PROCESSOR:
            self.PreProcessor[name] = callback
        elif callback_type == CallbackType.POST_PROCESSOR:
            self.PostProcessor[name] = callback
        elif callback_type == CallbackType.VISUALIZER:
            self.Visualizer[name] = callback
        elif callback_type == CallbackType.SOURCE:
            self.Source[name] = callback
        elif callback_type == CallbackType.COLLAT_INFER:
            self.CollatInfer[name] = callback

    def get_callback(
        self, name: str, callback_type: CallbackType, default: Optional[Callable] = None
    ) -> Union[Callable, Type]:
        """
        Retrieve a registered callback by name and type.

        Args:
            name: Name of the callback to retrieve
            callback_type: Type of callback to retrieve
            default: Default callback to return if not found (defaults to empty_callback)

        Returns:
            The registered callback, or default callback if not found

        Raises:
            ValueError: If callback_type is not a valid CallbackType
        """
        if not isinstance(callback_type, CallbackType):
            raise ValueError(
                f"Invalid callback type: {callback_type}. Must be a CallbackType enum."
            )

        if name not in self._callbacks[callback_type]:
            name = "base"

        callback = self._callbacks[callback_type][name]

        return callback

    def list_callbacks(
        self, callback_type: Optional[CallbackType] = None
    ) -> Dict[str, List[str]]:
        """
        List all registered callbacks, optionally filtered by type.

        Args:
            callback_type: Optional filter by callback type

        Returns:
            Dictionary mapping callback type names to lists of registered names
        """
        if callback_type:
            return {callback_type.value: list(self._callbacks[callback_type].keys())}

        return {
            cb_type.value: list(callbacks.keys())
            for cb_type, callbacks in self._callbacks.items()
        }

    def unregister_callback(self, name: str, callback_type: CallbackType) -> bool:
        """
        Unregister a callback by name and type.

        Args:
            name: Name of the callback to unregister
            callback_type: Type of the callback

        Returns:
            True if callback was found and removed, False otherwise
        """
        if not isinstance(callback_type, CallbackType):
            raise ValueError(
                f"Invalid callback type: {callback_type}. Must be a CallbackType enum."
            )

        if name in self._callbacks[callback_type]:
            del self._callbacks[callback_type][name]

            # Clean up legacy storage
            if (
                callback_type == CallbackType.PRE_PROCESSOR
                and name in self.PreProcessor
            ):
                del self.PreProcessor[name]
            elif (
                callback_type == CallbackType.POST_PROCESSOR
                and name in self.PostProcessor
            ):
                del self.PostProcessor[name]
            elif callback_type == CallbackType.VISUALIZER and name in self.Visualizer:
                del self.Visualizer[name]
            elif callback_type == CallbackType.SOURCE and name in self.Source:
                del self.Source[name]
            elif (
                callback_type == CallbackType.COLLAT_INFER and name in self.CollatInfer
            ):
                del self.CollatInfer[name]

            logger.debug(
                f"Unregistered callback '{name}' for type '{callback_type.value}'"
            )
            return True

        return False

    def has_callback(self, name: str, callback_type: CallbackType) -> bool:
        """
        Check if a callback is registered with the given name and type.

        Args:
            name: Name of the callback to check
            callback_type: Type of the callback

        Returns:
            True if callback exists, False otherwise
        """
        if not isinstance(callback_type, CallbackType):
            return False
        return name in self._callbacks[callback_type]

    # Convenient type-specific decorator methods
    def registryPreProcessor(self, name: str) -> Callable:
        """
        Convenient decorator for registering pre-processors.

        Args:
            name: Unique identifier for the pre-processor

        Returns:
            Decorator function

        Example:
            @CALLBACK_REGISTRY.registryPreProcessor("yolo_preprocess")
            def preprocess_func(image):
                return processed_image
        """
        return self.register(name, CallbackType.PRE_PROCESSOR)

    def registryPostProcessor(self, name: str) -> Callable:
        """
        Convenient decorator for registering post-processors.

        Args:
            name: Unique identifier for the post-processor

        Returns:
            Decorator function

        Example:
            @CALLBACK_REGISTRY.registryPostProcessor("yolo_postprocess")
            def postprocess_func(predictions):
                return detections
        """
        return self.register(name, CallbackType.POST_PROCESSOR)

    def registryVisualizer(self, name: str) -> Callable:
        """
        Convenient decorator for registering visualizers.

        Args:
            name: Unique identifier for the visualizer

        Returns:
            Decorator function

        Example:
            @CALLBACK_REGISTRY.registryVisualizer("bbox_visualizer")
            class BBoxVisualizer:
                def __init__(self, config):
                    pass
        """
        return self.register(name, CallbackType.VISUALIZER)

    def registrySource(self, name: str) -> Callable:
        """
        Convenient decorator for registering data sources.

        Args:
            name: Unique identifier for the data source

        Returns:
            Decorator function

        Example:
            @CALLBACK_REGISTRY.registrySource("video_source")
            class VideoSource:
                def __init__(self, path):
                    pass
        """
        return self.register(name, CallbackType.SOURCE)

    def registryCollatInfer(self, name: str) -> Callable:
        """
        Convenient decorator for registering collaborative inference components.

        Args:
            name: Unique identifier for the collaborative inference component

        Returns:
            Decorator function

        Example:
            @CALLBACK_REGISTRY.registryCollatInfer("yolov8det")
            def collat_infer_func(data):
                return results
        """
        return self.register(name, CallbackType.COLLAT_INFER)

    # Legacy methods for backward compatibility
    def getPreProcessor(self, name: str) -> Callable:
        """Legacy method - use get_callback instead"""
        return self.get_callback(name, CallbackType.PRE_PROCESSOR)

    def getPostProcessor(self, name: str) -> Callable:
        """Legacy method - use get_callback instead"""
        return self.get_callback(name, CallbackType.POST_PROCESSOR)

    def getVisualizer(self, name: str) -> Callable:
        """Legacy method - use get_callback instead"""
        return self.get_callback(name, CallbackType.VISUALIZER)

    def getSource(self, name: str) -> Callable:
        """Legacy method - use get_callback instead"""
        return self.get_callback(name, CallbackType.SOURCE)

    def getCollatInfer(self, name: str) -> Callable:
        """Legacy method - use get_callback instead"""
        return self.get_callback(name, CallbackType.COLLAT_INFER)


# Global registry instance
CALLBACK_REGISTRY = CallbackRegistry()


class InferenceEngine:
    def __init__(self, config: Config, callback_name: AnyStr) -> None:
        self.callback_name = callback_name
        self.config = config
        if self.config.command == "infer":
            self.infer = True
            if self.config.model.endswith(".hef"):
                self.infer_type = "hailo"
            elif self.config.model.endswith(".onnx"):
                self.infer_type = "onnx"
            else:
                raise ValueError(f"Unsupported model type: {self.config.model}")
        else:
            self.infer = False
        self.task_type = self.config.task_type

        self.callback_registry = CALLBACK_REGISTRY
        self.init_all()

    def init_all(self):
        self.init_model()
        self.init_source()
        self.init_preprocess()
        self.init_postprocess()
        self.init_visualization()
        self.init_callback()

    def init_model(self, model_file: Optional[Any] = None):
        if self.infer_type == "hailo":
            self.infer = HailoInference(self.config.model)
        elif self.infer_type == "onnx":
            self.infer = ONNXInference(self.config.model)

    def init_source(self, source: Optional[Any] = None):
        self.source = self.callback_registry.getSource(self.callback_name)(
            self.config.source
        )

    def init_preprocess(self, preprocess: Optional[Any] = None):
        self.preprocess = self.callback_registry.getPreProcessor(self.callback_name)(
            self.config.preprocess
        )

    def init_postprocess(self, postprocess: Optional[Any] = None):
        self.postprocess = self.callback_registry.getPostProcessor(self.callback_name)(
            self.config.postprocess
        )

    def init_visualization(self, visualization: Optional[Any] = None):
        # Create visualization config based on task type
        self.visualization = self.callback_registry.getVisualizer(self.callback_name)(
            self.config.visualization
        )

    def init_callback(self, callback: Optional[Any] = None):
        self.callback = self.infer.add_callback(
            self.callback_registry.getCollatInfer(self.callback_name)
        )

    @classmethod
    def load_from_config(cls, config):
        return cls(**config)

    def run(self):
        with self.source as source:
            for frame_idx, frame in enumerate(source):
                logger.debug(f"Processing frame {frame_idx}, shape: {frame.shape}")

                # Store original frame for visualization
                original_frame = frame.copy()

                # Preprocess frame
                preprocessed_frame = self.preprocess(frame)
                logger.debug(f"Preprocessed frame shape: {preprocessed_frame.shape}")

                # Run inference
                results = self.infer.as_process_inference(preprocessed_frame)

                # Postprocess results
                post_results = self.postprocess(
                    results, original_shape=original_frame.shape[:2]
                )

                # Visualize results
                try:
                    vis_image = self.visualization(original_frame, post_results)

                    # Save or display the visualization
                    output_path = f"output/output_frame_{frame_idx:04d}.jpg"
                    success = self.visualization.save(vis_image, output_path)
                    if success:
                        logger.debug(f"Saved visualization to {output_path}")

                    # Optionally display the image (comment out if running headless)
                    # self.visualization.show(vis_image, f"Frame {frame_idx}", wait_key=False)

                except Exception as e:
                    logger.error(f"Visualization error: {str(e)}")
                    import traceback

                    traceback.print_exc()

    def _call_callback(self, callback_type: CallbackType, frame: np.ndarray):
        if callback_type in self.callback_registry:
            self.callback_registry[callback_type](frame)


if __name__ == "__main__":
    print(CALLBACK_REGISTRY.getPreProcessor("yolov8det"))
