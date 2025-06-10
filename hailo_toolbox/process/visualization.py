import numpy as np


class VisualizationConfig:
    det: bool = False
    seg: bool = False
    kp: bool = False
    draw_bbox: bool = True
    draw_mask: bool = True
    draw_keypoints: bool = True
    draw_text: bool = True
    draw_score: bool = True
    draw_class: bool = True
    font_size: int = 1
    font_color: tuple = (0, 0, 0)
    font_thickness: int = 1
    font_family: str = "Arial"
    font_weight: str = "normal"
    font_style: str = "normal"
    font_underline: bool = False


class Visualization:
    def __init__(self, config: VisualizationConfig):
        pass

    def __call__(self, image: np.ndarray) -> np.ndarray:
        pass

    def save(self, image: np.ndarray, path: str):
        pass

    def show(self, image: np.ndarray):
        pass
