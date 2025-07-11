from pyglm.glm import scale, translate, vec3
from wgut.orbit_camera import OrbitCamera
from wgut.scene.ecs_explorer import ecs_explorer
from wgut.scene.loaders.obj import load_obj
from wgut.scene.performance_monitor import performance_monitor
from wgut.scene.render_gui_system import render_gui_system
from wgut.scene.render_system import (
    ActiveCamera,
    CameraComponent,
    Layer,
    MaterialComponent,
    MeshComponent,
    render_system,
)
from wgut.scene.renderer import Renderer
from wgut.scene.window_system import window_system
from wgut.scene.ecs import ECS
from wgut.scene.pbr_material import PbrMaterial
from wgut.scene.primitives.cube import cube
from wgut.scene.primitives.icosphere import icosphere
from wgut.scene.transform import Transform
from wgut.scene.direction_light import DirectionLight
from wgut.scene.ambiant_light import AmbiantLight

default_layer = Layer("default")

# TODO:
# - ECS gui
# - wireframe
# - gizmo test
# - frustum culling
# - ply, gltf loaders
# - drop shadow


def setup(ecs: ECS):
    mesh = cube()
    mesh = icosphere(3)
    bunny_mesh = load_obj("./models/bunny.obj")[0][0]
    wood_material = PbrMaterial(
        "./textures/Wood_025_basecolor.jpg",
        "./textures/Wood_025_normal.jpg",
        "./textures/Wood_025_roughness.jpg",
        0.0,
        (0.0, 0.0, 0.0),
        None,
    )
    dummy_material = PbrMaterial(
        "./textures/texel_checker.png",
        None,
        0.65,
        0.0,
        (0.0, 0.0, 0.0),
        None,
    )
    bunny_transform = Transform(translate(vec3(-2.5, -1.0, 0)) * scale(vec3(20)))  # type: ignore
    ball_transform = Transform(translate(vec3(2.5, 0, 0)))
    camera = OrbitCamera((0, 4, 6), (0, 0, 0), 45, 0.1, 100)
    ecs.spawn(
        [
            bunny_mesh,
            MaterialComponent(dummy_material),
            bunny_transform,
            default_layer,
        ],
        label="Bunny",
    )
    ecs.spawn(
        [
            MeshComponent(mesh),
            ball_transform,
            default_layer,
            MaterialComponent(wood_material),
        ],
        label="Ball",
    )
    id = ecs.spawn_group(load_obj("./models/f16_vertex_color/f16.obj"))
    ecs.add_component_to_group(id, Transform())
    ecs.add_component_to_group(id, default_layer)

    id = ecs.spawn_group(load_obj("./models/f16/f16.obj"))
    ecs.add_component_to_group(id, Transform(translate(vec3(0, 0, 2.5))))
    ecs.add_component_to_group(id, default_layer)

    ecs.spawn(DirectionLight.create(vec3(0, 0, -1), vec3(1.0), 3))
    ecs.spawn(AmbiantLight.create(vec3(1.0, 1.0, 1.0), 0.4))
    ecs.spawn([CameraComponent(camera), ActiveCamera()], label="Camera")


def process_event(ecs: ECS, event):
    cam: CameraComponent = ecs.query_one(CameraComponent)
    if isinstance(cam.camera, OrbitCamera):
        cam.camera.process_event(event)


renderer = Renderer(30000, 150000, 512, 2, (1024, 1024, 32), 256)


(
    ECS()
    .on("setup", setup)
    .on("window_event", process_event)
    .do(performance_monitor)
    .do(ecs_explorer)
    .do(
        render_system,
        renderer,
        [default_layer],
    )
    .do(render_gui_system)
    .do(window_system, "Hello ECS")
)
