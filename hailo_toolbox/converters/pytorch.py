"""
PyTorch model converter to ONNX format.
"""

import os
import torch
from typing import Any, Optional, Union, List, Tuple, Dict

from .base import BaseConverter
from ..utils import get_logger

# 创建模块的日志记录器
logger = get_logger(__name__)


class PyTorchConverter(BaseConverter):
    """
    Converter for PyTorch models to ONNX format.

    This class implements the conversion of PyTorch models to ONNX format
    with support for various configurations and optimizations.
    """

    def __init__(
        self,
        input_shape: Optional[Union[List[int], Tuple[int, ...]]] = None,
        opset_version: int = 11,
        output_dir: str = "converted_models",
        dynamic_axes: Optional[Dict] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the PyTorch converter.

        Args:
            input_shape: Input shape for the model (batch_size, channels, height, width).
            opset_version: ONNX opset version to use.
            output_dir: Directory to save converted models.
            dynamic_axes: Dictionary specifying dynamic axes for inputs/outputs.
            input_names: Names for the input tensors.
            output_names: Names for the output tensors.
            **kwargs: Additional parameters specific to PyTorch conversion.
        """
        super().__init__(input_shape, opset_version, output_dir, **kwargs)

        # Default input shape for image models if none provided
        if input_shape is None:
            self.input_shape = (1, 3, 224, 224)
            logger.info("使用默认输入形状: %s", self.input_shape)

        # Set input and output names
        self.input_names = input_names or ["input"]
        self.output_names = output_names or ["output"]

        # Set dynamic axes if provided
        self.dynamic_axes = dynamic_axes
        if self.dynamic_axes is None and self.input_shape[0] == 1:
            # Default dynamic batch size if batch dimension is 1
            self.dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }
            logger.debug("使用默认动态批次大小配置")

        logger.debug("已初始化PyTorch转换器")

    def load_model(self, model_path: str, **kwargs) -> Any:
        """
        Load a PyTorch model from a file.

        Args:
            model_path: Path to the PyTorch model (.pth, .pt, or directory).
            **kwargs: Additional loading parameters:
                - custom_load_func: Optional function to handle custom loading logic.
                - device: Device to load the model to ('cpu', 'cuda').
                - model_class: Optional model class for constructing before loading state dict.
                - model_args: Optional arguments for model class constructor.

        Returns:
            Loaded PyTorch model.
        """
        device = kwargs.get("device", "cpu")
        custom_load_func = kwargs.get("custom_load_func", None)
        model_class = kwargs.get("model_class", None)
        model_args = kwargs.get("model_args", {})

        logger.info("加载PyTorch模型: %s (device=%s)", model_path, device)

        if custom_load_func:
            logger.debug("使用自定义加载函数")
            model = custom_load_func(model_path, **kwargs)
        elif model_class:
            logger.debug("使用提供的模型类加载")
            model = model_class(**model_args)
            state_dict = torch.load(model_path, map_location=device)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
        else:
            logger.debug("使用torch.load直接加载模型")
            try:
                model = torch.load(model_path, map_location=device)
            except Exception as e:
                logger.error("模型加载失败: %s", e)
                raise ValueError(f"Failed to load model: {e}")

        model.to(device)
        model.eval()  # Set model to evaluation mode
        logger.info("PyTorch模型加载成功")
        return model

    def convert(
        self, model_path: str, output_name: Optional[str] = None, **kwargs
    ) -> str:
        """
        Convert a PyTorch model to ONNX format.

        Args:
            model_path: Path to the PyTorch model.
            output_name: Optional name for the output ONNX file.
            **kwargs: Additional conversion parameters:
                - input_sample: Optional input tensor sample.
                - simplify: Whether to simplify the ONNX model (default: True).
                - check_model: Whether to check the model after conversion (default: True).
                - Any parameter accepted by load_model.

        Returns:
            Path to the converted ONNX model.
        """
        # Generate output path
        if output_name is None:
            base_name = os.path.basename(model_path)
            name_no_ext = os.path.splitext(base_name)[0]
            output_name = f"{name_no_ext}.onnx"

        output_path = os.path.join(self.output_dir, output_name)
        logger.info("转换PyTorch模型到ONNX: %s -> %s", model_path, output_path)

        # Load the model
        model = self.load_model(model_path, **kwargs)

        # Create input sample if not provided
        input_sample = kwargs.get("input_sample", None)
        if input_sample is None:
            logger.debug("创建虚拟输入张量")
            input_sample = torch.randn(self.input_shape)
            device = next(model.parameters()).device
            input_sample = input_sample.to(device)

        # Export to ONNX
        logger.debug("开始ONNX导出: opset_version=%d", self.opset_version)
        try:
            torch.onnx.export(
                model,
                input_sample,
                output_path,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=self.input_names,
                output_names=self.output_names,
                dynamic_axes=self.dynamic_axes,
                verbose=kwargs.get("verbose", False),
            )

            logger.info("ONNX导出成功: %s", output_path)

            # Simplify if requested
            if kwargs.get("simplify", True):
                self._simplify_onnx(output_path)

            # Check the model if requested
            if kwargs.get("check_model", True):
                if self.validate_onnx(output_path):
                    logger.info("ONNX模型验证通过")
                else:
                    logger.warning("ONNX模型验证失败")

            return output_path

        except Exception as e:
            logger.error("ONNX转换失败: %s", e)
            raise RuntimeError(f"Failed to convert model to ONNX: {e}")

    def optimize_onnx(self, onnx_path: str) -> str:
        """
        Optimize the ONNX model using ONNX Runtime.

        Args:
            onnx_path: Path to the ONNX model.

        Returns:
            Path to the optimized ONNX model.
        """
        try:
            from onnxruntime.transformers import optimizer

            logger.info("使用ONNX Runtime优化模型: %s", onnx_path)

            # Get optimized model path
            opt_file = os.path.splitext(onnx_path)[0] + "_optimized.onnx"

            # Configure optimizer
            opt_options = optimizer.OptimizationOptions()

            # Create optimizer
            model_optimizer = optimizer.optimize_model(
                onnx_path,
                "bert",  # model type (can be adjusted based on your model)
                num_heads=12,  # default for BERT-base, adjust as needed
                hidden_size=768,  # default for BERT-base, adjust as needed
                optimization_options=opt_options,
            )

            # Save optimized model
            model_optimizer.save_model_to_file(opt_file)
            logger.info("优化后的模型已保存: %s", opt_file)
            return opt_file

        except ImportError:
            logger.warning("ONNX Runtime未安装，跳过优化")
            return onnx_path
        except Exception as e:
            logger.error("ONNX优化失败: %s", e)
            return onnx_path

    def _simplify_onnx(self, onnx_path: str) -> bool:
        """
        Simplify ONNX model using onnx-simplifier if available.

        Args:
            onnx_path: Path to the ONNX model.

        Returns:
            True if simplification succeeded, False otherwise.
        """
        try:
            import onnx
            from onnxsim import simplify

            logger.info("简化ONNX模型: %s", onnx_path)
            model = onnx.load(onnx_path)
            model_simp, check = simplify(model)

            if check:
                onnx.save(model_simp, onnx_path)
                logger.info("ONNX模型简化成功")
                return True
            else:
                logger.warning("ONNX模型简化过程完成，但模型可能无效")
                return False

        except ImportError:
            logger.warning("onnx-simplifier未安装，跳过简化")
            return False
        except Exception as e:
            logger.error("ONNX模型简化失败: %s", e)
            return False
