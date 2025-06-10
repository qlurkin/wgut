from dataclasses import dataclass
from pyglm.glm import (
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


@dataclass
class DirectionLight:
    color: vec3
    intensity: float

    def get_data(self, transformation_matrix: mat4) -> array[vec4]:
        direction = normalize(mat3(transformation_matrix) * vec3(0, 0, -1))  # type: ignore
        return array(vec4(direction, 0.0), vec4(self.color, self.intensity))  # type: ignore

    @staticmethod
    def create_transform(direction: vec3) -> mat4:
        v = vec3(0, 0, -1)
        direction = normalize(direction)

        cross_product = cross(v, direction)
        dot_product = dot(v, direction)

        if length(cross_product) < 1e-6:
            if dot_product > 0:
                return identity(mat4)
            else:
                orthogonal = vec3(1, 0, 0) if abs(v.x) < 0.9 else vec3(0, 1, 0)
                axis = normalize(cross(v, orthogonal))
                return rotate(pi(), axis)

        k = normalize(cross_product)
        K = mat3(0, -k.z, k.y, k.z, 0, -k.x, -k.y, k.x, 0)

        i = identity(mat3)
        R = i + K + K * K * ((1 - dot_product) / (length(cross_product) ** 2))
        return mat4(R)

    @staticmethod
    def create(direction: vec3, color: vec3, intensity: float = 1.0) -> list:
        light = DirectionLight(color, intensity)
        transform = DirectionLight.create_transform(direction)
        return [light, transform]
