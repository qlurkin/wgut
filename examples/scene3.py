# imgui imports first
from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiRenderer
from wgpu import GPUTexture
from wgut import Window, get_device
from wgut.scene.primitives.icosphere import icosphere
from wgut.scene.renderer3 import Renderer, Material
from wgut.scene.transform import Transform
from wgut.orbit_camera import OrbitCamera
import numpy as np


class MyApp(Window):
    def setup(self):
        self.set_title("Hello Scene")

        self.mesh = icosphere(3)
        self.material1 = Material(
            albedo="./textures/Substance_Graph_BaseColor.jpg",
            normal="./textures/Substance_Graph_Normal.jpg",
            roughness="./textures/Substance_Graph_Roughness.jpg",
            occlusion="./textures/Substance_Graph_AmbientOcclusion.jpg",
        )
        self.material2 = Material(
            albedo="./textures/Wood_025_basecolor.jpg",
            normal="./textures/Wood_025_normal.jpg",
            roughness="./textures/Wood_025_roughness.jpg",
        )
        self.material3 = Material(albedo=(1.0, 0.0, 0.0, 1.0))

        self.lights = np.array(
            [
                [0, 0, 0, 0],  # means ambiant light
                [1, 1, 1, 0.4],  # color and intensity
                [0, 0, -1, 0],  # means direction light
                [1, 1, 1, 3],  # color and intensity
            ],
            dtype=np.float32,
        )

        self.renderer = Renderer(10000, 50000, 2, 1024, 32)

        self.camera = OrbitCamera((6, 4, 5), (0, 0, 0), 45, 0.1, 100)

        self.imgui_renderer = ImguiRenderer(
            get_device(), self.get_canvas(), self.get_texture_format()
        )

        self.imgui_renderer.set_gui(self.gui)

        self.frame_time = 0

    def update(self, delta_time: float):
        self.imgui_renderer.backend.io.delta_time = delta_time
        self.frame_time = delta_time

    def render(self, screen: GPUTexture):
        translation0 = Transform().set_translation([0, 0, 2.5])
        translation1 = Transform().set_translation([2.5, 0, 0])
        translation2 = Transform().set_translation([-2.5, 0, 0])

        self.renderer.clear_color(screen, (0.9, 0.9, 0.9, 1.0))
        self.renderer.clear_depth()
        view_matrix, proj_matrix = self.camera.get_matrices(
            screen.width / screen.height
        )

        camera_position = np.hstack([self.camera.get_position(), [1.0]]).astype(
            np.float32
        )
        camera_matrix = np.array(proj_matrix @ view_matrix, dtype=np.float32)
        self.renderer.begin_frame(
            screen,
            camera_matrix,
            camera_position,
            self.lights,
        )

        self.renderer.add_mesh(self.mesh, translation0, self.material1)
        self.renderer.add_mesh(self.mesh, translation1, self.material2)
        self.renderer.add_mesh(self.mesh, translation2, self.material3)

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
        self.camera.process_event(event)


MyApp().run()
