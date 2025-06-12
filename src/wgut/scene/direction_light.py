from dataclasses import dataclass
from pyglm.glm import (
    acos,
    array,
    dot,
    identity,
    length,
    mat3,
    mat4,
    normalize,
    pi,
    rotate,
    vec3,
    vec4,
    cross,
)

from wgut.scene.light import LightComponent
from wgut.scene.transform import Transform


@dataclass
class DirectionLight:
    color: vec3
    intensity: float

    def get_data(self, transformation_matrix: mat4) -> array[vec4]:
        direction = normalize(mat3(transformation_matrix) * vec3(0, 0, -1))  # type: ignore
        return array(vec4(direction, 0.0), vec4(self.color, self.intensity))  # type: ignore

    @staticmethod
    def create_transform(direction: vec3) -> Transform:
        v = vec3(0, 0, -1)
        direction = normalize(direction)

        cross_product = cross(v, direction)
        dot_product = dot(v, direction)

        if length(cross_product) < 1e-6:
            if dot_product > 0:
                return Transform(identity(mat4))
            else:
                axis = vec3(1, 0, 0)
                return Transform(rotate(pi(), axis))

        k = normalize(cross_product)

        return Transform(rotate(acos(dot_product), k))

    @staticmethod
    def create(direction: vec3, color: vec3, intensity: float = 1.0) -> list:
        light = DirectionLight(color, intensity)
        transform = DirectionLight.create_transform(direction)
        return [LightComponent(light), transform]
