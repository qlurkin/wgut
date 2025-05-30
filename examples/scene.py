# imgui imports first
from imgui_bundle import imgui
from pyglm.glm import translate, vec3
from wgpu.utils.imgui import ImguiRenderer
from wgpu import GPUTexture
from wgut import Window, get_device
from wgut.scene.primitives.icosphere import icosphere
from wgut.scene.renderer import Renderer
from wgut.scene.transform import Transform
from wgut.scene.pbr_material import PbrMaterial
from wgut.scene.basic_color_material import BasicColorMaterial
from wgut.orbit_camera import OrbitCamera


class MyApp(Window):
    def setup(self):
        self.set_title("Hello Scene")

        self.mesh = icosphere(3)
        self.material1 = PbrMaterial(
            "./textures/Substance_Graph_BaseColor.jpg",
            "./textures/Substance_Graph_Normal.jpg",
            "./textures/Substance_Graph_Roughness.jpg",
            0.0,
            (0.0, 0.0, 0.0),
            "./textures/Substance_Graph_AmbientOcclusion.jpg",
        )
        self.material2 = PbrMaterial(
            "./textures/Wood_025_basecolor.jpg",
            "./textures/Wood_025_normal.jpg",
            "./textures/Wood_025_roughness.jpg",
            0.0,
            (0.0, 0.0, 0.0),
            None,
        )
        self.material3 = BasicColorMaterial((1.0, 0.0, 0.0))

        self.renderer = Renderer(10000, 50000, 128, (1024, 1024, 7), 48)

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
        translation0 = Transform(translate(vec3(0, 0, 2.5)))
        translation1 = Transform(translate(vec3(2.5, 0, 0)))
        translation2 = Transform(translate(vec3(-2.5, 0, 0)))

        self.renderer.begin_frame()
        self.renderer.add_mesh(self.mesh, translation0, self.material1)
        self.renderer.add_mesh(self.mesh, translation1, self.material2)
        self.renderer.add_mesh(self.mesh, translation2, self.material3)
        self.renderer.end_frame(screen, self.camera)

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
