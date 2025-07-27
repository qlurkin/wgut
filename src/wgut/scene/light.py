from dataclasses import dataclass
from typing import Protocol


class Light(Protocol):
    def get_data(self, transformation_matrix: mat4) -> array[vec4]: ...


@dataclass
class LightComponent:
    light: Light
