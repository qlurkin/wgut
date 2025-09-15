from dataclasses import dataclass
from typing import Protocol
import numpy.typing as npt


class Light(Protocol):
    def get_data(self, transformation_matrix: npt.NDArray) -> npt.NDArray: ...


@dataclass
class LightComponent:
    light: Light

    def __str__(self):
        return str(self.light)
