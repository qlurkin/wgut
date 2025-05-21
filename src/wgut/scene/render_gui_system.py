from typing import Callable

from imgui_bundle import imgui
from wgpu import GPUTexture
from wgpu.utils.imgui import ImguiRenderer

from wgut.scene.ecs import ECS, System
from wgut import get_device
from wgut.scene.window_system import WindowSystemApp


def render_gui_system(gui_func: Callable[[ECS], imgui.ImDrawData]) -> System:
    imgui_renderer: ImguiRenderer | None = None

    def render(_ecs: ECS, _screen: GPUTexture):
        if imgui_renderer is not None:
            imgui_renderer.render()

    def update(ecs: ECS, delta_time: float):
        if imgui_renderer is not None:
            imgui_renderer.backend.io.delta_time = delta_time

    def setup(ecs: ECS):
        nonlocal imgui_renderer
        (window,) = ecs.query_one([WindowSystemApp])
        imgui_renderer = ImguiRenderer(
            get_device(), window.get_canvas(), window.get_texture_format()
        )

        def gui() -> imgui.ImDrawData:
            return gui_func(ecs)

        imgui_renderer.set_gui(gui)

        ecs.on("render", render)
        ecs.on("update", update)

    return setup
