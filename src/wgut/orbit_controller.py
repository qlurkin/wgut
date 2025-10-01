import numpy as np
from numpy.typing import NDArray
from pygfx import (
    Camera,
    Event,
    PointerEvent,
    WgpuRenderer,
    WheelEvent,
)
from math import pi

from wgut.ecs import ECS


def cartesian_to_spherical(pos: NDArray) -> tuple[float, float, float]:
    r = np.linalg.norm(pos)
    theta = np.arccos(pos[1] / r) if r != 0 else 0.0  # Inclinaison [0, π]
    phi = np.arctan2(pos[2], pos[0])  # Azimut [-π, π]
    return r.astype(float), theta, phi


def spherical_to_cartesian(r: float, theta: float, phi: float) -> NDArray:
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.cos(theta)
    z = r * np.sin(theta) * np.sin(phi)
    return np.array([x, y, z])


class OrbitController:
    def __init__(
        self,
        ecs: ECS,
        camera: Camera,
        target: tuple[float, float, float],
    ):
        self.__camera = camera
        self.__target = np.array(target)
        centered_position: NDArray = self.__camera.local.position - self.__target

        self.__radius, self.__theta, self.__phi = cartesian_to_spherical(
            centered_position
        )

        self.__moving = False
        self.__move_start = (0.0, 0.0)
        self.__pointer_start = (0.0, 0.0)

        # ecs.on("pygfx_event", self.process_event)
        ecs.on("update", self.update)

        ecs.dispatch("call_with_renderer", self.register_to_renderer)

    def register_to_renderer(self, renderer: WgpuRenderer):
        renderer.add_event_handler(
            self.process_event,
            "pointer_down",
            "wheel",
            "pointer_up",
            "pointer_move",
            "key_down",
            "key_up",
            "before_render",
        )

    def process_event(self, event: Event):
        event_type = event.type

        if isinstance(event, WheelEvent):
            dw = event.dy
            delta_radius = dw * 0.01
            self.__radius += delta_radius
            if self.__radius < 0.1:
                self.__radius = 0.1

        if isinstance(event, PointerEvent):
            if event_type == "pointer_down":
                self.__moving = True
                self.__move_start = (self.__theta, self.__phi)
                self.__pointer_start = (event.x, event.y)

            if self.__moving:
                if event_type == "pointer_up":
                    self.__moving = False
                if event_type == "pointer_move":
                    delta_x = event.x - self.__pointer_start[0]
                    delta_y = event.y - self.__pointer_start[1]
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

    def update(self, ecs: ECS, delta_time: float):
        self.__camera.local.position = spherical_to_cartesian(
            self.__radius, self.__theta, self.__phi
        )
        self.__camera.look_at(self.__target)  # type: ignore
