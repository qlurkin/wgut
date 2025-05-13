import numpy as np
import numpy.typing as npt


class UniformPbrMaterial:
    def __init__(self, color: npt.NDArray, roughness: float, metalicity: float):
        self.__data = np.hstack([color, [roughness], [metalicity]], dtype=np.float32)

    @staticmethod
    def get_fragment() -> str:
        return ""

    def get_data(self) -> npt.NDArray:
        return self.__data

    @staticmethod
    def get_data_size() -> int:
        # TODO: Check alignement
        return 24
