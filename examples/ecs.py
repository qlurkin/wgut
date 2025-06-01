from imgui_bundle import imgui
from wgut.orbit_camera import OrbitCamera
from wgut.scene.render_gui_system import render_gui_system
from wgut.scene.render_system import (
    ActiveCamera,
    CameraComponent,
    Layer,
    MaterialComponent,
    RenderStat,
    render_system,
)
from wgut.scene.renderer import Renderer
from wgut.scene.window_system import window_system
from wgut.scene.ecs import ECS, EntityNotFound
from wgut.scene.pbr_material import PbrMaterial
from wgut.scene.primitives.icosphere import icosphere
from wgut.scene.primitives.cube import cube
from wgut.scene.primitives.torus import torus
from wgut.scene.primitives.cone import cone
from wgut.scene.primitives.cylinder import cylinder
from wgut.scene.transform import Transform

default_layer = Layer()


def setup(ecs: ECS):
    mesh = icosphere(3)
    # mesh = cone()
    mesh = torus()
    # mesh = cube()
    mesh = cylinder()
    material = PbrMaterial(
        "./textures/Wood_025_basecolor.jpg",
        "./textures/Wood_025_normal.jpg",
        "./textures/Wood_025_roughness.jpg",
        0.0,
        (0.0, 0.0, 0.0),
        None,
    )
    transform = Transform()
    camera = OrbitCamera((6, 4, 5), (0, 0, 0), 45, 0.1, 100)
    ecs.spawn(
        [mesh, MaterialComponent(material), transform, default_layer], label="Ball"
    )
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


renderer = Renderer(30000, 90000, 128, (1024, 1024, 7), 48)


(
    ECS()
    .on("setup", setup)
    .on("window_event", process_event)
    .on("render_gui", gui)
    .do(render_system, renderer, [default_layer])
    .do(render_gui_system)
    .do(window_system, "Hello ECS")
)
