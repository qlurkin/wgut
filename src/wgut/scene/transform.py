import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from typing import Self


class Transform:
    def __init__(self, matrix: npt.NDArray | None = None, parent: Self | None = None):
        if matrix is None:
            matrix = np.identity(4)
        self.__matrix = matrix
        self.__children = []
        self.__parent = parent

    def from_translation_rotation_scale(
        self,
        translation: npt.NDArray,
        rotation: npt.NDArray,
        scale: npt.NDArray,
    ):
        R_with_scale = rotation * scale
        res = np.hstack([R_with_scale, translation])
        return np.vstack([res, np.array([0, 0, 0, 1])])

    def set_translation(self, translation: npt.NDArray):
        R = self.get_rotation_matrix()
        S = self.get_scale()
        self.from_translation_rotation_scale(translation, R, S)

    def set_rotation(self, rotation_matrix: npt.NDArray):
        T = self.get_translation()
        S = self.get_scale()
        self.from_translation_rotation_scale(T, rotation_matrix, S)

    def set_scale(self, scale: npt.NDArray):
        T = self.get_translation()
        R = self.get_rotation_matrix()
        self.from_translation_rotation_scale(T, R, scale)

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

    def add_child(self, child: Self):
        self.__children.append(child)

    def get_parent(self) -> Self | None:
        return self.__parent

    def get_children(self) -> list[Self]:
        return list(self.__children)
