from pygfx import (
    GridHelper,
    PerspectiveCamera,
    Background,
    AmbientLight,
    DirectionalLight,
    Mesh,
    MeshStandardMaterial,
    TransformGizmo,
    sphere_geometry,
    load_mesh,
    OrbitController as OC,
)

from wgut import (
    OrbitController,
    ActiveCamera,
    SceneObject,
    ecs_explorer,
    render_system,
    window_system,
    ECS,
    create_texture,
    render_gui_system,
    performance_monitor,
)


# TODO:
# - better import paths
# - docstrings and typing
# - test wireframe
# - gizmo test
# - test drop shadow
# - ECS gui


def setup(ecs: ECS, _):
    def test_event(event):
        print("EVENT:", event.type)

    ball = Mesh(
        sphere_geometry(1),
        MeshStandardMaterial(
            map=create_texture("./textures/Wood_025_basecolor.jpg"),
            normal_map=create_texture("./textures/Wood_025_normal.jpg"),
            roughness_map=create_texture("./textures/Wood_025_roughness.jpg"),
            pick_write=True,
        ),
    )
    ecs.spawn([SceneObject(ball)], label="Ball")
    ball.local.x -= 1
    ball.add_event_handler(test_event, "pointer_down")

    bunny = load_mesh("./models/bunny.obj")[0]

    bunny.local.x += 1
    bunny.local.y -= 0.5
    bunny.local.scale = 10

    bunny.material = MeshStandardMaterial(
        map=create_texture("./textures/texel_checker.png")
    )

    ecs.spawn([SceneObject(bunny)], label="Bunny")
    gizmo = TransformGizmo(object=bunny)

    def setup_gizmo(renderer):
        gizmo.add_default_event_handlers(renderer, camera)

    ecs.dispatch("call_with_renderer", setup_gizmo)
    ecs.spawn([SceneObject(gizmo, layer=1)])

    ecs.spawn([SceneObject(Background.from_color((0.9, 0.9, 0.9)))])

    ecs.spawn([SceneObject(AmbientLight(intensity=0.6))])
    ecs.spawn([SceneObject(DirectionalLight())])

    camera = PerspectiveCamera(70, 16 / 9)
    camera.local.position = (2, 2, 2)
    camera.look_at((0, 0, 0))

    ecs.spawn([SceneObject(camera), ActiveCamera()])
    ecs.spawn([SceneObject(GridHelper())])

    OrbitController(ecs, camera, target=(0, 0, 0))

    def call_renderer(renderer):
        controller = OC(camera, target=(0, 0, 0), register_events=renderer)

    # ecs.dispatch("call_with_renderer", call_renderer)


(
    ECS()
    .on("setup", setup)
    .do(performance_monitor)
    .do(ecs_explorer)
    .do(render_system)
    .do(render_gui_system)
    .do(window_system, "Hello ECS")
)
