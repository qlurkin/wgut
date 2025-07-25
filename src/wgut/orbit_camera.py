import numpy as np
from pyglm.glm import acos, length, lookAt, mat4, perspective, vec3
from math import atan2, pi


def cartesian_to_spherical(pos: vec3) -> tuple[float, float, float]:
    r = length(pos)
    theta = acos(pos[1] / r) if r != 0 else 0.0  # Inclinaison [0, π]
    phi = atan2(pos[2], pos[0])  # Azimut [-π, π]
    return r, theta, phi


def spherical_to_cartesian(r: float, theta: float, phi: float) -> vec3:
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.cos(theta)
    z = r * np.sin(theta) * np.sin(phi)
    return vec3([x, y, z])


class OrbitCamera:
    def __init__(
        self,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
        fovy: float,
        near: float,
        far: float,
    ):
        self.__target = vec3(target)
        centered_position = vec3(position) - self.__target

        self.__radius, self.__theta, self.__phi = cartesian_to_spherical(
            centered_position
        )

        self.__fovy = fovy
        self.__near = near
        self.__far = far
        self.__moving = False
        self.__move_start = (0.0, 0.0)
        self.__pointer_start = (0.0, 0.0)

    def get_matrices(self, ratio: float) -> tuple[mat4, mat4]:
        position = self.get_position()
        # view_matrix = cm.look_at(position, self.__target, [0, 1, 0])
        view_matrix = lookAt(position, self.__target, vec3(0, 1, 0))
        # proj_matrix = cm.perspective(self.__fovy, ratio, self.__near, self.__far)
        proj_matrix = perspective(self.__fovy, ratio, self.__near, self.__far)

        return view_matrix, proj_matrix

    def get_position(self) -> vec3:
        return self.__target + spherical_to_cartesian(
            self.__radius, self.__theta, self.__phi
        )

    def process_event(self, event) -> bool:
        dirty = False

        event_type = event["event_type"]

        if event_type == "wheel":
            dw = event["dy"]
            delta_radius = dw * 0.01
            self.__radius += delta_radius
            if self.__radius < 0.1:
                self.__radius = 0.1

        if event_type == "pointer_down":
            self.__moving = True
            self.__move_start = (self.__theta, self.__phi)
            self.__pointer_start = (event["x"], event["y"])

        if self.__moving:
            if event_type == "pointer_up":
                self.__moving = False
            if event_type == "pointer_move":
                delta_x = event["x"] - self.__pointer_start[0]
                delta_y = event["y"] - self.__pointer_start[1]
                delta_theta = delta_y * -0.01
                delta_phi = delta_x * 0.01
                self.__theta = self.__move_start[0] + delta_theta
                if self.__theta < 0.1:
                    self.__theta = 0.1
                if self.__theta > pi - 0.1:
                    self.__theta = pi - 0.1
                self.__phi = self.__move_start[1] + delta_phi
                if self.__phi > pi:
                    self.__phi -= 2 * pi
                if self.__phi < -pi:
                    self.__phi += 2 * pi
            dirty = True
        return dirty
