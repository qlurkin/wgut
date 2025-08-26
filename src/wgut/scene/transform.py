from typing import Self
from imgui_bundle import imgui
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation


class Transform:
    def __init__(
        self,
        matrix: npt.NDArray | None = None,
    ):
        if matrix is None:
            matrix = np.identity(4, dtype=np.float32)
        self.__matrix = np.array(matrix, dtype=np.float32)

    def from_translation_rotation_scale(
        self,
        translation: npt.ArrayLike,
        rotation: npt.ArrayLike,
        scale: npt.ArrayLike,
    ) -> Self:
        R_with_scale = np.array(rotation) * np.array(scale)
        res = np.hstack([R_with_scale, np.array([translation]).T])
        self.__matrix = np.vstack([res, np.array([0, 0, 0, 1])], dtype=np.float32)
        return self

    def set_translation(self, translation: npt.ArrayLike) -> Self:
        R = self.get_rotation_matrix()
        S = self.get_scale()
        self.from_translation_rotation_scale(translation, R, S)
        return self

    def set_rotation(self, rotation_matrix: npt.ArrayLike) -> Self:
        T = self.get_translation()
        S = self.get_scale()
        self.from_translation_rotation_scale(T, rotation_matrix, S)
        return self

    def set_scale(self, scale: npt.ArrayLike) -> Self:
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
