from imgui_bundle import imgui
from wgpu import GPUTexture
from wgpu.utils.imgui import ImguiRenderer

from wgut.scene.ecs import ECS
from wgut import get_device
from wgut.scene.window_system import WindowSystemApp
from wgut.window import Window


def render_gui_system(ecs: ECS):
    imgui_renderer: ImguiRenderer | None = None

    def render(_ecs: ECS, _screen: GPUTexture):
        if imgui_renderer is not None:
            imgui_renderer.render()

    def update(_ecs: ECS, delta_time: float):
        if imgui_renderer is not None:
            imgui_renderer.backend.io.delta_time = delta_time

    def setup(ecs: ECS, window: Window):
        nonlocal imgui_renderer
        imgui_renderer = ImguiRenderer(
            get_device(), window.get_canvas(), window.get_texture_format()
        )

        def gui() -> imgui.ImDrawData:
            imgui.new_frame()
            ecs.dispatch("render_gui")
            imgui.end_frame()
            imgui.render()
            return imgui.get_draw_data()

        imgui_renderer.set_gui(gui)

        ecs.on("after_render", render)
        ecs.on("update", update)

    ecs.on("setup", setup)
