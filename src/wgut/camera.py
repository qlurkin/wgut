from typing import Protocol

from pyglm.glm import mat4, vec3


class Camera(Protocol):
    def get_matrices(self, ratio: float) -> tuple[mat4, mat4]: ...
    def get_position(self) -> vec3: ...
