from typing import Protocol

from numpy.typing import NDArray


class Camera(Protocol):
    def get_matrices(self, ratio: float) -> tuple[NDArray, NDArray]: ...
