from typing import Self
from imgui_bundle import imgui
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from wgut.cgmath import rotation_matrix_from_axis_and_angle


class Transform:
    def __init__(
        self,
        matrix: npt.NDArray | None = None,
    ):
        if matrix is None:
            matrix = np.identity(4, dtype=np.float32)
        self.__matrix = np.array(matrix, dtype=np.float32)
        self.__parent: Self | None = None
        self.__children: frozenset[Self] = frozenset()

    def add_child(self, child: Self):
        child.__parent = self
        self.__children |= frozenset([child])

    def get_parent(self) -> Self | None:
        return self.__parent

    def remove_child(self, child):
        child.__parent = None
        self.__children -= frozenset([child])

    def get_children(self) -> frozenset[Self]:
        return self.__children

    def from_translation_rotation_scale(
        self,
        translation: npt.ArrayLike,
        rotation: npt.ArrayLike,
        scale: npt.ArrayLike,
    ) -> Self:
        scale = np.array(scale)
        if scale.size == 1:
            scale = np.array([scale, scale, scale])
        if scale.size == 3:
            scale = np.diag(scale)
        R_with_scale = np.array(rotation) * scale
        res = np.hstack([R_with_scale, np.array([translation]).T])
        self.__matrix = np.vstack([res, np.array([0, 0, 0, 1])], dtype=np.float32)
        return self

    def set_translation(self, translation: npt.ArrayLike) -> Self:
        R = self.get_rotation_matrix()
        S = self.get_scale()
        self.from_translation_rotation_scale(translation, R, S)
        return self

    def set_rotation_matrix(self, rotation_matrix: npt.ArrayLike) -> Self:
        T = self.get_translation()
        S = self.get_scale()
        self.from_translation_rotation_scale(T, rotation_matrix, S)
        return self

    def set_rotation_angle_and_axis(self, angle: float, axis: npt.ArrayLike) -> Self:
        T = self.get_translation()
        S = self.get_scale()
        R = rotation_matrix_from_axis_and_angle(axis, angle)
        self.from_translation_rotation_scale(T, R, S)
        return self

    def set_scale(self, scale: npt.ArrayLike) -> Self:
        T = self.get_translation()
        R = self.get_rotation_matrix()
        self.from_translation_rotation_scale(T, R, scale)
        return self

    def get_matrix(self) -> npt.NDArray:
        if self.__parent is not None:
            return self.__parent.get_matrix() @ self.__matrix
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

    def __str__(self):
        return "Transform"

    def ecs_explorer_gui(self):
        if imgui.collapsing_header("translation"):
            translation = self.get_translation()
            imgui.input_float("x", translation[0])
            imgui.input_float("y", translation[1])
            imgui.input_float("z", translation[2])

        imgui.collapsing_header("rotation")
        imgui.collapsing_header("scale")
