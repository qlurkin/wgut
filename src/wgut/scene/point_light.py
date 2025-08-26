from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
from wgut.scene.light import LightComponent
from wgut.scene.transform import Transform
from wgut.cgmath import vec


@dataclass
class PointLight:
    color: npt.NDArray
    intensity: float

    def get_data(self, transformation_matrix: npt.NDArray) -> npt.NDArray:
        position = transformation_matrix * vec([0, 0, 0, 1])
        position = position / position[3]
        return np.hstack([position, self.color, [self.intensity]], dtype=np.float32)

    @staticmethod
    def create_transform(position: npt.NDArray) -> Transform:
        return Transform(
            np.array(
                [
                    [1, 0, 0, position[0]],
                    [0, 1, 0, position[1]],
                    [0, 0, 1, position[2]],
                    [0, 0, 0, 1],
                ]
            )
        )

    @staticmethod
    def create(
        position: npt.NDArray, color: npt.NDArray, intensity: float = 1.0
    ) -> list:
        light = PointLight(color, intensity)
        transform = PointLight.create_transform(position)
        return [LightComponent(light), transform]
