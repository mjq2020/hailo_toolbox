import os
from .base import BaseConverter
from typing import Optional, Union, List, Dict, Tuple, Any
from pathlib import Path
from logging import getLogger
from glob import glob
import cv2
import numpy as np
from hailo_sdk_client.model_translator.exceptions import (
    MisspellNodeError,
    ParsingWithRecommendationException,
    UnsupportedModelError,
)

logger = getLogger(__file__)


IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif", "*.webp"]


class HailoConverter(BaseConverter):
    def __init__(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        framework: Optional[str] = None,
        hw_arch: Optional[str] = None,
        har: Optional[str] = None,
        script: Optional[Union[str, Path]] = None,
        calibration_dataset: Optional[Union[str, Path]] = None,
        calibration_dataset_size: int = 100,
        height: int = 224,
        width: int = 224,
        rgb: bool = True,
    ):
        super().__init__(model_path, output_path, framework, hw_arch, har)
        self.script = script
        self.calibration_dataset = calibration_dataset
        self.calibration_dataset_size = calibration_dataset_size
        self.set_calibration_dataset(calibration_dataset, height, width, rgb)

    def set_calibration_dataset(
        self,
        calibration_dataset: Optional[Union[str, Path]] = None,
        height: int = 224,
        width: int = 224,
        rgb: bool = True,
    ):
        self.calibration_dataset = []
        if calibration_dataset:
            image_paths = []
            for ext in IMAGE_EXTENSIONS:
                image_paths.extend(
                    glob(os.path.join(calibration_dataset, ext), recursive=True)
                )
            image_paths = image_paths[: self.calibration_dataset_size]
            for img_path in image_paths:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (height, width))
                if rgb:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.calibration_dataset.append(img)
        else:
            if rgb:
                self.calibration_dataset = np.random.randint(
                    0, 255, (self.calibration_dataset_size, height, width, 3)
                )
            else:
                self.calibration_dataset = np.random.randint(
                    0, 255, (self.calibration_dataset_size, height, width, 1)
                )

    def convert(self):
        pass

    def onnx_to_har(
        self,
        onnx_file: Optional[Union[str, Path, bytes]] = None,
        model_name: Optional[str] = None,
        start_node_names: Optional[List[str]] = None,
        end_node_names: Optional[List[str]] = None,
        net_input_shapes: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
        augmented_path: Optional[str] = None,
        disable_shape_inference: bool = False,
        disable_rt_metadata_extraction: bool = False,
        net_input_format: Optional[str] = None,
        **kwargs,
    ):
        if onnx_file:
            self.set_onnx_file(onnx_file)

        try:
            self.runner.translate_onnx_model(
                model=self.onnx_file,
                net_name=model_name,
                start_node_names=start_node_names,
                end_node_names=end_node_names,
                net_input_shapes=net_input_shapes,
                augmented_path=augmented_path,
                disable_shape_inference=disable_shape_inference,
                disable_rt_metadata_extraction=disable_rt_metadata_extraction,
                net_input_format=net_input_format,
                **kwargs,
            )
        except ParsingWithRecommendationException as e:
            end_nodes = str(e).split(": ")[-1].split(", ")
            self.runner.translate_onnx_model(
                model=self.onnx_file,
                net_name=model_name,
                start_node_names=start_node_names,
                end_node_names=end_nodes,
                net_input_shapes=net_input_shapes,
                augmented_path=augmented_path,
                disable_shape_inference=disable_shape_inference,
                disable_rt_metadata_extraction=disable_rt_metadata_extraction,
                net_input_format=net_input_format,
                **kwargs,
            )
        else:
            raise e

        self.runner.save_har(self.onnx_file.with_suffix(".har"))

        self.set_har_file(self.onnx_file.with_suffix(".har"))

    def tf_to_har(
        self,
        tf_file: Optional[Union[str, Path, bytes]] = None,
        model_name: Optional[str] = None,
        start_node_names: Optional[List[str]] = None,
        end_node_names: Optional[List[str]] = None,
        net_input_shapes: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
        **kwargs,
    ):
        if tf_file:
            self.set_tf_file(tf_file)
        self.runner.translate_tf_model(
            model=self.tf_file,
            net_name=model_name,
            start_node_names=start_node_names,
            end_node_names=end_node_names,
            net_input_shapes=net_input_shapes,
        )
        self.runner.save_har(self.tf_file.with_suffix(".har"))
        self.set_har_file(self.tf_file.with_suffix(".har"))

    def visualize_har(
        self, har_file: Optional[Union[str, Path, bytes]] = None, verbose: bool = False
    ):
        """
        Visualize the HAR file.
        """
        if har_file:
            self.set_har_file(har_file)

        svg_file = self.har_file.with_suffix(".svg")

        self._get_hailo_nn().visualize(svg_file.absolute().as_posix(), verbose=verbose)

    def visualize_hn(
        self, hn_file: Optional[Union[str, Path, bytes]] = None, verbose: bool = False
    ):
        """
        Visualize the HN file.
        """
        if hn_file:
            self.set_hn_file(hn_file)

        svg_file = self.hn_file.with_suffix(".svg")

        self._get_hailo_nn().visualize(svg_file.absolute().as_posix(), verbose=verbose)

    def add_model_script(
        self, script: Optional[Union[str, Path]] = None, append: bool = False
    ):
        if script:
            self.runner.load_model_script(script, append)
        else:
            self.runner.load_model_script(self.script, append)

    def parse(self):
        self.runner.load_model_script()

    def optimize(self):
        assert hasattr(self, "calibration_dataset"), "Calibration dataset is not set"
        self.runner.optimize(self.calibration_dataset)
        self.runner.save_har(
            self.model_path.with_name(self.model_path.stem + "_optimized.har")
        )

    def compile(self):
        hef_model = self.runner.compile()
        self.wirte_file(self.model_path.with_suffix(".hef"), hef_model)

    def dump_model_info(self):
        pass

    def wirte_file(self, file_path: str, content: Any):
        with open(file_path, "w") as f:
            f.write(content)


if __name__ == "__main__":
    converter = HailoConverter(
        model_path="",
        output_path="",
        framework="",
    )
