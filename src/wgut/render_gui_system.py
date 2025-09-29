from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiRenderer

from wgut.ecs import ECS
from wgut import get_device
from wgut.window import Window


def render_gui_system(ecs: ECS):
    def setup(ecs: ECS, window: Window):
        imgui_renderer = ImguiRenderer(get_device(), window.get_canvas())

        def render(_ecs: ECS):
            imgui_renderer.render()

        def update(_ecs: ECS, delta_time: float):
            imgui_renderer.backend.io.delta_time = delta_time

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
