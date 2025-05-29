from typing import Self
from pyglm.glm import mat4, identity, mat3, inverseTranspose


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
        return inverseTranspose(mat3(self.__matrix))

    def add_child(self, child: Self):
        self.__children.append(child)

    def get_parent(self) -> Self | None:
        return self.__parent

    def get_children(self) -> list[Self]:
        return list(self.__children)
