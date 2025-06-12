from dataclasses import dataclass
from typing import Protocol

from pyglm.glm import array, mat4, vec4


class Light(Protocol):
    def get_data(self, transformation_matrix: mat4) -> array[vec4]: ...


@dataclass
class LightComponent:
    light: Light
