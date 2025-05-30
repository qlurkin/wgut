import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from typing import Self
from .mesh import Mesh


class Transform:
    def __init__(
        self,
        matrix: npt.NDArray | None = None,
        parent: Self | None = None,
        mesh: Mesh | None = None,
    ):
        if matrix is None:
            matrix = np.identity(4, dtype=np.float32)
        self.__matrix = np.array(matrix, dtype=np.float32)
        self.__children = []
        self.__parent = parent
        self.__mesh = mesh

    def from_translation_rotation_scale(
        self,
        translation: npt.NDArray,
        rotation: npt.NDArray,
        scale: npt.NDArray,
    ) -> Self:
        R_with_scale = rotation * scale
        res = np.hstack([R_with_scale, translation])
        self.__matrix = np.vstack([res, np.array([0, 0, 0, 1])], dtype=np.float32)
        return self

    def set_translation(self, translation: npt.NDArray) -> Self:
        R = self.get_rotation_matrix()
        S = self.get_scale()
        self.from_translation_rotation_scale(translation, R, S)
        return self

    def set_rotation(self, rotation_matrix: npt.NDArray) -> Self:
        T = self.get_translation()
        S = self.get_scale()
        self.from_translation_rotation_scale(T, rotation_matrix, S)
        return self

    def set_scale(self, scale: npt.NDArray) -> Self:
        T = self.get_translation()
        R = self.get_rotation_matrix()
        self.from_translation_rotation_scale(T, R, scale)
        return self

    def get_matrix(self) -> npt.NDArray:
        return self.__matrix

    def get_translation(self) -> npt.NDArray:
        return self.__matrix[:3, 3]

    def get_scale(self) -> npt.NDArray:
        M = self.__matrix

        scale_x = np.linalg.norm(M[:3, 0])
        scale_y = np.linalg.norm(M[:3, 1])
        scale_z = np.linalg.norm(M[:3, 2])

        return np.array([scale_x, scale_y, scale_z])

    def get_rotation_matrix(self) -> npt.NDArray:
        M = self.__matrix

        R_with_scale = M[:3, :3]

        return R_with_scale / self.get_scale()

    def get_rotation_euler(self) -> npt.NDArray:
        R_mat = self.get_rotation_matrix()
        return Rotation.from_matrix(R_mat).as_euler("xyz")

    def get_rotation_quaternion(self) -> npt.NDArray:
        R_mat = self.get_rotation_matrix()
        return Rotation.from_matrix(R_mat).as_quat()

    def get_normal_matrix(self) -> npt.NDArray:
        M3 = self.__matrix[:3, :3]
        return np.linalg.inv(M3).T

    def add_child(self, child: Self):
        self.__children.append(child)

    def get_parent(self) -> Self | None:
        return self.__parent

    def get_children(self) -> list[Self]:
        return list(self.__children)

    def get_mesh(self) -> Mesh | None:
        return self.__mesh
