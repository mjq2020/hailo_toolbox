from enum import Enum
from abc import ABC, abstractmethod
import numpy as np


class CallbackType(Enum):
    AFTER_INFER = "after_infer"
    AFTER_POSTPROCESS = "after_postprocess"
    AFTER_PREPROCESS = "after_preprocess"
    BEFORE_INFER = "before_infer"
    BEFORE_POSTPROCESS = "before_postprocess"
    BEFORE_PREPROCESS = "before_preprocess"


class BaseCallback(ABC):
    def __init__(self, callback_type: CallbackType):
        self.callback_type = callback_type

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        pass
