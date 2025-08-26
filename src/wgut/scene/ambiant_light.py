from dataclasses import dataclass

from wgut.cgmath import vec4
from wgut.scene.light import LightComponent
from wgut.scene.transform import Transform

import numpy as np
import numpy.typing as npt


@dataclass
class AmbiantLight:
    color: npt.NDArray
    intensity: float

    def __init__(self, color: npt.ArrayLike, intensity: float):
        self.color = np.array(color)
        self.intensity = intensity

    def get_data(self, transformation_matrix: npt.ArrayLike) -> npt.NDArray:
        return np.array([vec4(0.0), vec4(self.color, self.intensity)])

    @staticmethod
    def create_transform() -> Transform:
        return Transform()

    @staticmethod
    def create(color: npt.ArrayLike, intensity: float = 1.0) -> list:
        light = AmbiantLight(color, intensity)
        transform = AmbiantLight.create_transform()
        return [LightComponent(light), transform]
