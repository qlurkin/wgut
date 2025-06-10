from dataclasses import dataclass
from pyglm.glm import (
    array,
    mat4,
    vec3,
    vec4,
)


@dataclass
class AmbiantLight:
    color: vec3
    intensity: float

    def get_data(self, transformation_matrix: mat4) -> array[vec4]:
        return array(vec4(self.direction, 0.0), vec4(self.color, self.intensity))  # type: ignore

    @staticmethod
    def create_transform() -> mat4:
        return mat4()

    @staticmethod
    def create(color: vec3, intensity: float = 1.0) -> list:
        light = AmbiantLight(color, intensity)
        transform = AmbiantLight.create_transform()
        return [light, transform]
