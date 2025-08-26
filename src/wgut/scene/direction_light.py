from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
from wgut.scene.light import LightComponent
from wgut.scene.transform import Transform
from wgut.cgmath import normalize, vec3, length


@dataclass
class DirectionLight:
    color: npt.NDArray
    intensity: float

    def __init__(self, color: npt.ArrayLike, intensity: float):
        self.color = np.array(color)
        self.intensity = intensity

    def get_data(self, transformation_matrix: npt.NDArray) -> npt.NDArray:
        direction = normalize(transformation_matrix[:3, :3] @ vec3([0, 0, -1]))
        return np.array(
            [np.hstack([direction, [0.0]]), np.hstack([self.color, [self.intensity]])],
            dtype=np.float32,
        )

    @staticmethod
    def create_transform(direction: npt.ArrayLike) -> Transform:
        v = vec3([0, 0, -1])
        direction = normalize(direction)

        cross_product = np.cross(v, direction)
        dot_product = np.dot(v, direction)

        if length(cross_product) < 1e-6:
            if dot_product > 0:
                return Transform()
            else:
                axis = vec3([1, 0, 0])
                return Transform().set_rotation_angle_and_axis(np.pi, axis)

        k = normalize(cross_product)

        return Transform().set_rotation_angle_and_axis(np.acos(dot_product), k)

    @staticmethod
    def create(
        direction: npt.ArrayLike, color: npt.ArrayLike, intensity: float = 1.0
    ) -> list:
        light = DirectionLight(color, intensity)
        transform = DirectionLight.create_transform(direction)
        return [LightComponent(light), transform]
