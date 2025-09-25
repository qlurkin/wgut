from wgut.orbit_camera import OrbitCamera
from wgut.scene.ecs_explorer import ecs_explorer
from wgut.scene.loaders.obj import load_obj
from wgut.scene.performance_monitor import performance_monitor
from wgut.scene.render_gui_system import render_gui_system
from wgut.scene.render_system import (
    ActiveCamera,
    CameraComponent,
    MaterialComponent,
    MeshComponent,
    render_system,
)
from wgut.scene.renderer import Renderer, Material
from wgut.scene.window_system import window_system
from wgut.scene.ecs import ECS
from wgut.scene.primitives.cube import cube
from wgut.scene.primitives.icosphere import icosphere
from wgut.scene.transform import Transform
from wgut.scene.direction_light import DirectionLight
from wgut.scene.ambiant_light import AmbiantLight


# TODO:
# - better import paths
# - docstrings and typing
# - wireframe
# - gizmo test
# - frustum culling
# - ply, gltf loaders
# - drop shadow
# - ECS gui


def setup(ecs: ECS):
    mesh = cube()
    mesh = icosphere(3)
    bunny_mesh = load_obj("./models/bunny.obj")[0][0]
    wood_material = Material(
        albedo="./textures/Wood_025_basecolor.jpg",
        normal="./textures/Wood_025_normal.jpg",
        roughness="./textures/Wood_025_roughness.jpg",
    )
    dummy_material = Material(
        albedo="./textures/texel_checker.png",
        roughness=0.65,
    )
    bunny_transform = Transform().set_translation([-2.5, -1.0, 0]).set_scale(20)
    ball_transform = Transform().set_translation([2.5, 0, 0])
    camera = OrbitCamera((0, 4, 6), (0, 0, 0), 45, 0.1, 100)
    ecs.spawn(
        [
            bunny_mesh,
            MaterialComponent(dummy_material),
            bunny_transform,
        ],
        label="Bunny",
    )
    ecs.spawn(
        [
            MeshComponent(mesh),
            ball_transform,
            MaterialComponent(wood_material),
        ],
        label="Ball",
    )
    id = ecs.spawn_group(load_obj("./models/f16_vertex_color/f16.obj"))

    id = ecs.spawn_group(load_obj("./models/f16/f16.obj"))
    print(ecs[id][Transform].set_translation([0, 0, 2.5]))

    ecs.spawn(DirectionLight.create((0, 0, -1), (1, 1, 1), 3))
    ecs.spawn(AmbiantLight.create((1, 1, 1), 0.4))
    ecs.spawn([CameraComponent(camera), ActiveCamera()], label="Camera")


def process_event(ecs: ECS, event):
    cam: CameraComponent = ecs.query_one(CameraComponent)
    if isinstance(cam.camera, OrbitCamera):
        cam.camera.process_event(event)


renderer = Renderer(30000, 150000, 4, 512, 32)

(
    ECS()
    .on("setup", setup)
    .on("window_event", process_event)
    .do(performance_monitor)
    .do(ecs_explorer)
    .do(
        render_system,
        renderer,
    )
    .do(render_gui_system)
    .do(window_system, "Hello ECS")
)
