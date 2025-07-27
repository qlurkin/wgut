from dataclasses import dataclass

from wgut.scene.light import LightComponent
from wgut.scene.transform import Transform


@dataclass
class PointLight:
    color: vec3
    intensity: float

    def get_data(self, transformation_matrix: mat4) -> array[vec4]:
        position = transformation_matrix * vec4(0, 0, 0, 1)
        position = position / position[3]
        return array(position, vec4(self.color, self.intensity))  # type: ignore

    @staticmethod
    def create_transform(position: vec3) -> Transform:
        return Transform(
            mat4(
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                position[0],
                position[1],
                position[2],
                1,
            )
        )

    @staticmethod
    def create(position: vec3, color: vec3, intensity: float = 1.0) -> list:
        light = PointLight(color, intensity)
        transform = PointLight.create_transform(position)
        return [LightComponent(light), transform]
