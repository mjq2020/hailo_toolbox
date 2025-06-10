"""
ONNX model inference engine implementation.
"""

import os
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from .base import BaseInferenceEngine, InferenceResult


class ONNXInference(BaseInferenceEngine):
    """
    Inference engine for ONNX models using ONNX Runtime.
    """

    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ONNX inference engine.

        Args:
            model_path: Path to the ONNX model file.
            config: Configuration dictionary containing:
                - providers: List of execution providers (default: ["CPUExecutionProvider"]).
                - execution_mode: Execution mode (default: "SEQUENTIAL").
                - graph_optimization_level: Graph optimization level (default: "ORT_ENABLE_ALL").
                - inter_op_num_threads: Number of threads to use for inter-op parallelism.
                - intra_op_num_threads: Number of threads to use for intra-op parallelism.
                - input_name: Name of the input tensor (will be auto-detected if not provided).
                - output_names: Names of the output tensors (will be auto-detected if not provided).
                - input_shape: Shape of the input tensor (will be auto-detected if not provided).
                - model_name: Name of the model (default: derived from model_path).
        """
        # Default config values
        default_config = {
            "providers": ["CPUExecutionProvider"],
            "execution_mode": "SEQUENTIAL",
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "model_name": os.path.splitext(os.path.basename(model_path))[0],
        }

        # Merge with provided config
        merged_config = {**default_config, **(config or {})}

        super().__init__(model_path, merged_config)

        # ONNX Runtime specific attributes
        self.session = None
        self.input_name = self.config.get("input_name")
        self.output_names = self.config.get("output_names")
        self.input_shape = self.config.get("input_shape")

        # Execution provider settings
        self.providers = self.config["providers"]
        self.execution_mode = self.config["execution_mode"]
        self.graph_optimization_level = self.config["graph_optimization_level"]
        self.inter_op_num_threads = self.config.get("inter_op_num_threads")
        self.intra_op_num_threads = self.config.get("intra_op_num_threads")

        # Auto-detect CUDA availability and add CUDAExecutionProvider if available
        if "CUDAExecutionProvider" not in self.providers and self._is_cuda_available():
            self.providers.insert(0, "CUDAExecutionProvider")

    def _is_cuda_available(self) -> bool:
        """
        Check if CUDA is available for ONNX Runtime.

        Returns:
            True if CUDA is available, False otherwise.
        """
        try:
            import onnxruntime as ort

            return "CUDAExecutionProvider" in ort.get_available_providers()
        except:
            return False

    def load(self) -> bool:
        """
        Load the ONNX model.

        Returns:
            True if the model was loaded successfully, False otherwise.
        """
        try:
            import onnxruntime as ort

            # Check if model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")

            # Set up session options
            session_options = ort.SessionOptions()

            # Set execution mode
            if self.execution_mode == "SEQUENTIAL":
                session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            elif self.execution_mode == "PARALLEL":
                session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

            # Set graph optimization level
            if self.graph_optimization_level == "ORT_DISABLE_ALL":
                session_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                )
            elif self.graph_optimization_level == "ORT_ENABLE_BASIC":
                session_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                )
            elif self.graph_optimization_level == "ORT_ENABLE_EXTENDED":
                session_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
                )
            elif self.graph_optimization_level == "ORT_ENABLE_ALL":
                session_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )

            # Set threading options
            if self.inter_op_num_threads is not None:
                session_options.inter_op_num_threads = self.inter_op_num_threads
            if self.intra_op_num_threads is not None:
                session_options.intra_op_num_threads = self.intra_op_num_threads

            # Create inference session
            self.session = ort.InferenceSession(
                self.model_path, sess_options=session_options, providers=self.providers
            )

            # Auto-detect input and output info if not provided
            if self.input_name is None:
                self.input_name = self.session.get_inputs()[0].name

            if self.output_names is None:
                self.output_names = [
                    output.name for output in self.session.get_outputs()
                ]

            if self.input_shape is None:
                self.input_shape = self.session.get_inputs()[0].shape

            # Update config with detected values
            self.config["input_name"] = self.input_name
            self.config["output_names"] = self.output_names
            self.config["input_shape"] = self.input_shape

            self.is_loaded = True
            return True

        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            self.is_loaded = False
            return False

    def unload(self) -> None:
        """Unload the ONNX model and free resources."""
        self.session = None
        self.is_loaded = False

    def infer(self, input_data: Dict[str, np.ndarray]) -> InferenceResult:
        """
        Run inference on the input data.

        Args:
            input_data: Dictionary mapping input names to numpy arrays.

        Returns:
            InferenceResult containing the inference results.
        """
        if not self.is_loaded or self.session is None:
            return InferenceResult(
                success=False,
                model_name=self.model_name,
                raw_outputs={},
                metadata={"error": "Model not loaded"},
            )

        try:
            # Start timing
            start_time = time.time()

            # Run inference
            outputs = self.session.run(self.output_names, input_data)

            # Calculate inference time
            inference_time_ms = (time.time() - start_time) * 1000

            # Create output dictionary
            raw_outputs = {
                name: output for name, output in zip(self.output_names, outputs)
            }

            # Create result
            result = InferenceResult(
                success=True,
                model_name=self.model_name,
                raw_outputs=raw_outputs,
                input_data=input_data,
                inference_time_ms=inference_time_ms,
            )

            return result

        except Exception as e:
            print(f"Inference error: {e}")
            return InferenceResult(
                success=False,
                model_name=self.model_name,
                raw_outputs={},
                input_data=input_data,
                metadata={"error": str(e)},
            )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded ONNX model.

        Returns:
            Dictionary with model information.
        """
        info = super().get_model_info()

        if self.is_loaded and self.session is not None:
            # Add ONNX specific information
            inputs = []
            for i in self.session.get_inputs():
                inputs.append({"name": i.name, "shape": i.shape, "type": str(i.type)})

            outputs = []
            for o in self.session.get_outputs():
                outputs.append({"name": o.name, "shape": o.shape, "type": str(o.type)})

            info.update(
                {
                    "inputs": inputs,
                    "outputs": outputs,
                    "providers": self.session.get_providers(),
                    "input_shape": self.input_shape,
                    "output_names": self.output_names,
                }
            )

        return info
