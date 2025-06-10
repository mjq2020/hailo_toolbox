from typing import Optional, Union, Dict, Any, AnyStr
import numpy as np
from hailo_tools.utils.config import Config
from hailo_tools.cli.config import parse_args

from hailo_tools.inference.hailo_engine import HailoInference
from hailo_tools.inference.onnx_engine import ONNXInference
from hailo_tools.sources import create_source

from hailo_tools.process import ImagePreprocessor
from hailo_tools.process.postprocessor_det import YOLOv8DetPostprocessor
from hailo_tools.process.postprocessor_seg import YOLOv8SegPostprocessor
from hailo_tools.process.postprocessor_kp import YOLOv8KpPostprocessor
from hailo_tools.process.visualization import Visualization
from hailo_tools.process.callback import BaseCallback, CallbackType


class Inference:
    def __init__(self, **kwargs) -> None:
        self.config = Config(kwargs)
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

        self.callback_registry = {}

    def init_all(self):
        if self.infer:
            self.init_model()
            self.init_source()
            self.init_preprocess()
            self.init_postprocess()
        else:
            self.init_convert()

    def init_convert(self):
        pass

    def init_model(self, model_file: Optional[Any] = None):
        if self.infer_type == "hailo":
            self.infer = HailoInference(self.config.model)
            self.infer = HailoInference(self.config.model)
        elif self.infer_type == "onnx":
            self.infer = ONNXInference(self.config.model)

    def init_source(self, source: Optional[Any] = None):
        self.source = create_source(self.config.source)

    def init_preprocess(self, preprocess: Optional[Any] = None):
        self.preprocess = ImagePreprocessor(self.config.preprocess)

    def init_postprocess(self, postprocess: Optional[Any] = None):
        if self.infer_type == "det":
            self.postprocess = YOLOv8DetPostprocessor()
        elif self.infer_type == "seg":
            self.postprocess = YOLOv8SegPostprocessor()
        elif self.infer_type == "kp":
            self.postprocess = YOLOv8KpPostprocessor()

    def init_visualization(self, visualization: Optional[Any] = None):
        self.visualization = Visualization(self.config.visualization)

    def add_callback(self, callback: BaseCallback):
        self.callback_registry[callback.callback_type] = callback

    @classmethod
    def load_from_config(cls, config: Dict[AnyStr, Any]):
        return cls(**config)

    def run(self):
        with self.source as source:
            with self.infer as infer:
                for frame in source:
                    frame = self.preprocess(frame)
                    infer.infer(frame)
                    frame = self.postprocess(frame)
                    self.visualization(frame)

    def _call_callback(self, callback_type: CallbackType, frame: np.ndarray):
        if callback_type in self.callback_registry:
            self.callback_registry[callback_type](frame)


def main():
    args = parse_args()
    print(vars(args))
    inference = Inference(**vars(args))


if __name__ == "__main__":
    main()
