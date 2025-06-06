from imgui_bundle import imgui
from pyglm.glm import array, scale, vec3, vec4
from wgpu import GPUBuffer
from wgut.orbit_camera import OrbitCamera
from wgut.scene.particles import Particles
from wgut.scene.render_gui_system import render_gui_system
from wgut.scene.render_system import (
    ActiveCamera,
    CameraComponent,
    Layer,
    MaterialComponent,
    MeshComponent,
    RenderStat,
    render_system,
)
from wgut.scene.renderer import Renderer
from wgut.scene.window_system import window_system
from wgut.scene.ecs import ECS, EntityNotFound
from wgut.scene.basic_color_material import BasicColorMaterial
from wgut.scene.primitives.icosphere import icosphere
from wgut.scene.transform import Transform

default_layer = Layer("default")


def setup(ecs: ECS):
    mesh = icosphere(3)

    def callback(ecs: ECS, buffer: GPUBuffer, delta_time: float):
        pass

    particles = Particles(
        mesh,
        array(
            vec4(1, 0, 1, 0), vec4(-1, 0, 1, 0), vec4(1, 0, -1, 0), vec4(-1, 0, -1, 0)
        ),
        callback,
    )

    material = BasicColorMaterial((1.0, 0.0, 0.0))

    ecs.spawn(
        [
            MaterialComponent(material),
            MeshComponent(particles),
            Transform(scale(vec3(0.2))),
            default_layer,
        ]
    )

    camera = OrbitCamera((6, 4, 5), (0, 0, 0), 45, 0.1, 100)

    ecs.spawn([CameraComponent(camera), ActiveCamera()], label="Camera")


def process_event(ecs: ECS, event):
    cam: CameraComponent = ecs.query_one(CameraComponent)
    if isinstance(cam.camera, OrbitCamera):
        cam.camera.process_event(event)


def gui(ecs: ECS):
    try:
        r_stat: RenderStat = ecs.query_one(RenderStat)
        stat = r_stat.stats[default_layer]
        imgui.begin("Render Stats", None)
        imgui.text(f"Render Time: {stat['time']:.5f}s")
        imgui.text(f"Draw count: {stat['draw']}")
        imgui.text(f"Mesh count: {stat['mesh']}")
        imgui.text(f"Triangle count: {stat['triangle']}")
        imgui.text(f"Vertex count: {stat['vertex']}")
        imgui.end()
    except EntityNotFound:
        pass


renderer = Renderer(30000, 150000, 512, (1024, 1024, 32), 256)


(
    ECS()
    .on("setup", setup)
    .on("window_event", process_event)
    .on("render_gui", gui)
    .do(
        render_system,
        renderer,
        [default_layer],
    )
    .do(render_gui_system)
    .do(window_system, "Hello Particles")
)
