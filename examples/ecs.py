from imgui_bundle import imgui
from pyglm.glm import scale, translate, vec3
from wgut.orbit_camera import OrbitCamera
from wgut.scene.loaders.obj import load_obj
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
from wgut.scene.primitives.cube import cube
from wgut.scene.primitives.icosphere import icosphere
from wgut.scene.transform import Transform

default_layer = Layer("default")

# TODO:
# - spawn light
# - better perf gui
# - wireframe
# - gizmo test
# - frustum culling
# - obj, ply, gltf loaders
# - spawn_group
# - particle systems
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
    camera = OrbitCamera((6, 4, 5), (0, 0, 0), 45, 0.1, 100)
    ecs.spawn(
        [bunny_mesh, MaterialComponent(dummy_material), bunny_transform, default_layer],
        label="Bunny",
    )
    ecs.spawn(
        [mesh, ball_transform, default_layer, MaterialComponent(wood_material)],
        label="Ball",
    )
    f16 = load_obj("./models/f16_vertex_color/f16.obj")
    for entt in f16:
        entt.append(Transform())
        entt.append(default_layer)
        ecs.spawn(entt)
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


renderer = Renderer(20000, 100000, 256, (1024, 1024, 32), 128)


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
    .do(window_system, "Hello ECS")
)
