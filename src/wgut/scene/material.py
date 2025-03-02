import numpy as np
import numpy.typing as npt


class Material:
    def __init__(self, color: npt.NDArray, roughness: float, metalicity: float):
        self.__data = np.hstack([color, [roughness], [metalicity]], dtype=np.float32)

    def get_data(self):
        return self.__data
