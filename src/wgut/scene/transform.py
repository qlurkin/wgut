from typing import Self
from imgui_bundle import imgui
from pyglm.glm import mat4, identity, mat3, inverseTranspose, vec3, vec4


class Transform:
    def __init__(
        self,
        matrix: mat4 | None = None,
        parent: Self | None = None,
    ):
        if matrix is None:
            matrix = identity(mat4)
        self.__matrix = matrix
        self.__children = []
        self.__parent = parent

    def get_matrix(self) -> mat4:
        return self.__matrix

    def get_normal_matrix(self) -> mat3:
        return inverseTranspose(mat3(self.__matrix))  # type: ignore

    def add_child(self, child: Self):
        self.__children.append(child)

    def get_parent(self) -> Self | None:
        return self.__parent

    def get_children(self) -> list[Self]:
        return list(self.__children)

    def get_translation(self) -> vec3:
        return self.__matrix * vec3(0, 0, 0)  # type: ignore

    def __str__(self):
        return "Transform"

    def ecs_explorer_gui(self):
        if imgui.collapsing_header("translation"):
            translation = self.get_translation()
            imgui.input_float("x", translation.x)
            imgui.input_float("y", translation.y)
            imgui.input_float("z", translation.z)

        imgui.collapsing_header("rotation")
        imgui.collapsing_header("scale")
