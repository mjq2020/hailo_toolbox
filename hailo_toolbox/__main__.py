from typing import Optional, Union, Dict, Any, AnyStr
import numpy as np
from hailo_toolbox.utils.config import Config
from hailo_toolbox.cli.config import parse_args

from hailo_toolbox.inference.hailo_engine import HailoInference
from hailo_toolbox.inference.onnx_engine import ONNXInference
from hailo_toolbox.sources import create_source

from hailo_toolbox.process import ImagePreprocessor
from hailo_toolbox.process.postprocessor.postprocessor_det import YOLOv8DetPostprocessor
from hailo_toolbox.process.postprocessor.postprocessor_seg import (
    YOLOv8SegPostprocessor,
    YOLOv5SegPostprocessor,
)
from hailo_toolbox.process.postprocessor.postprocessor_pe import YOLOv8PosePostprocessor
from hailo_toolbox.process.visualization.visualization import (
    Visualization,
    VisualizationConfig,
    KeypointVisualization,
    SegmentationVisualization,
)
from hailo_toolbox.process.callback import BaseCallback, CallbackType
from hailo_toolbox.process.callback import yolov8_det_callback
from hailo_toolbox.utils.logging import get_logger

logger = get_logger(__name__)


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
        self.task_type = self.config.task_type

        self.callback_registry = {}

    def init_all(self):
        if self.infer:
            self.init_model()
            self.init_source()
            self.init_preprocess()
            self.init_postprocess()
            self.init_visualization()
        else:
            self.init_convert()

    def init_convert(self):
        pass

    def init_model(self, model_file: Optional[Any] = None):
        if self.infer_type == "hailo":
            self.infer = HailoInference(self.config.model)
            # self.infer.add_callback(yolov8_det_callback)
        elif self.infer_type == "onnx":
            self.infer = ONNXInference(self.config.model)

    def init_source(self, source: Optional[Any] = None):
        self.source = create_source(self.config.source)

    def init_preprocess(self, preprocess: Optional[Any] = None):
        self.preprocess = ImagePreprocessor(self.config.preprocess)

    def init_postprocess(self, postprocess: Optional[Any] = None):
        if self.task_type == "det":
            self.postprocess = YOLOv8DetPostprocessor()
        elif self.task_type == "seg":
            self.postprocess = YOLOv5SegPostprocessor()
        elif self.task_type == "kp":
            self.postprocess = YOLOv8PosePostprocessor()

    def init_visualization(self, visualization: Optional[Any] = None):
        # Create visualization config based on task type
        vis_config = VisualizationConfig()

        # Enable appropriate visualization flags based on task type
        if self.task_type == "det":
            vis_config.det = True
            vis_config.draw_bbox = True
            vis_config.draw_text = True
            vis_config.draw_score = True
            vis_config.draw_class = True
        elif self.task_type == "seg":
            vis_config.seg = True
            vis_config.draw_mask = True
            vis_config.draw_bbox = True
            vis_config.draw_text = True
            vis_config.draw_score = True
            vis_config.draw_class = True
            vis_config.mask_alpha = 0.5
        elif self.task_type == "kp":
            vis_config.kp = True
            vis_config.draw_keypoints = True
            vis_config.draw_skeleton = True
            vis_config.draw_person_bbox = True
            vis_config.draw_text = True
            vis_config.draw_score = True

        self.visualization = KeypointVisualization(vis_config)

    def add_callback(self, callback: BaseCallback):
        self.callback_registry[callback.callback_type] = callback

    @classmethod
    def load_from_config(cls, config: Dict[AnyStr, Any]):
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
                # print(f"Inference results keys: {results.keys()}")

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


def main():
    args = parse_args()
    print(vars(args))
    inference = Inference(**vars(args))
    inference.init_all()
    inference.run()


if __name__ == "__main__":
    main()
