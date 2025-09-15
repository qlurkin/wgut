from wgpu import GPUTexture
from wgpu.utils.imgui import ImguiRenderer

from wgut.scene.ecs import ECS
from wgut import get_device
from wgut.scene.window_system import WindowSystemApp


def render_gui_system(ecs: ECS):
    imgui_renderer: ImguiRenderer | None = None

    def render(_ecs: ECS, _screen: GPUTexture):
        if imgui_renderer is not None:
            imgui_renderer.render()

    def setup(ecs: ECS):
        nonlocal imgui_renderer
        window: WindowSystemApp = ecs.query_one(WindowSystemApp)
        imgui_renderer = ImguiRenderer(
            get_device(), window.get_canvas(), window.get_texture_format()
        )

        def gui():
            ecs.dispatch("render_gui")

        imgui_renderer.set_gui(gui)

        ecs.on("after_render", render)

    ecs.on("setup", setup)
