from wgut import (
    Window,
    create_canvas,
    create_texture,
)
from pygfx import (
    Background,
    OrbitController,
    PerspectiveCamera,
    Scene,
    WgpuRenderer,
    MeshStandardMaterial,
    Mesh,
    AmbientLight,
    DirectionalLight,
    sphere_geometry,
)

canvas = create_canvas(title="Hello Icosphere", max_fps=60)


class MyApp(Window):
    def setup(self):
        self.set_title("Hello Icosphere")

        self.scene = Scene()
        self.renderer = WgpuRenderer(target=self.get_canvas())

        icosphere = Mesh(
            geometry=sphere_geometry(),
            material=MeshStandardMaterial(
                map=create_texture("./textures/Wood_025_basecolor.jpg"),
                normal_map=create_texture("./textures/Wood_025_normal.jpg"),
                roughness_map=create_texture("./textures/Wood_025_roughness.jpg"),
            ),
        )

        self.scene.add(icosphere)

        self.scene.add(Background.from_color((0.9, 0.9, 0.9)))

        self.scene.add(AmbientLight(intensity=0.6))
        self.scene.add(DirectionalLight())

        self.camera = PerspectiveCamera(fov=70)
        self.camera.local.position = (2, 2, 2)
        controller = OrbitController(self.camera, target=(0, 0, 0))
        controller.register_events(self.renderer)

    def render(self):
        self.renderer.clear(all=True)
        self.renderer.render(self.scene, self.camera)


MyApp(canvas).run()
