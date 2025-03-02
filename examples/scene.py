# imgui imports first
from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiRenderer
from wgpu import GPUTexture
from wgut import Window, load_file, get_device
import numpy as np
import wgut.cgmath as cm
from wgut.scene.material import Material
from wgut.scene.primitives.icosphere import icosphere
from wgut.scene.renderer import Renderer
from wgut.scene.transform import Transform


class MyApp(Window):
    def setup(self):
        self.set_title("Hello Scene")

        self.mesh = icosphere(3)

        self.renderer = Renderer(load_file("scene.wgsl"), 1000, 10000)

        self.imgui_renderer = ImguiRenderer(
            get_device(), self.get_canvas(), self.get_texture_format()
        )

        self.imgui_renderer.set_gui(self.gui)

        self.frame_time = 0

    def update(self, delta_time: float):
        self.imgui_renderer.backend.io.delta_time = delta_time
        self.frame_time = delta_time

    def render(self, screen: GPUTexture):
        width, height, _ = screen.size
        view_matrix = cm.look_at([6, 4, 5], [0, 0, 0], [0, 1, 0])
        proj_matrix = cm.perspective(45, width / height, 0.1, 100)
        camera_matrix = np.array(proj_matrix @ view_matrix, dtype=np.float32)

        translation1 = Transform().set_translation(
            np.array([[2.5, 0, 0]], dtype=np.float32).T
        )
        translation2 = Transform().set_translation(
            np.array([[-2.5, 0, 0]], dtype=np.float32).T
        )

        material = Material(np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32), 0.5, 0.0)

        self.renderer.begin_frame(screen, camera_matrix)
        self.renderer.add_mesh(self.mesh, Transform(), material)
        self.renderer.add_mesh(self.mesh, translation1, material)
        self.renderer.add_mesh(self.mesh, translation2, material)
        self.renderer.end_frame()

        self.imgui_renderer.render()

    def gui(self) -> imgui.ImDrawData:
        stat = self.renderer.get_frame_stat()
        imgui.new_frame()
        imgui.begin("Scene", None)
        imgui.text(f"Frame Time: {self.frame_time:.5f}s")
        if stat is not None:
            imgui.text(f"Draw count: {stat['draw']}")
            imgui.text(f"Mesh count: {stat['mesh']}")
            imgui.text(f"Triangle count: {stat['triangle']}")
            imgui.text(f"Vertex count: {stat['vertex']}")
        imgui.end()
        imgui.end_frame()
        imgui.render()
        return imgui.get_draw_data()

    def process_event(self, event):
        pass


MyApp().run()
