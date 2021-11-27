"""Module with abs execution class"""

import abc

import numpy as np


class BaseExecutor(abc.ABC):
    @abc.abstractmethod
    def execute(self, image: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _get_model_result(self, image: np.ndarray) -> np.ndarray:
        pass
