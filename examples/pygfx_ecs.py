from pygfx import (
    GridHelper,
    PerspectiveCamera,
    Background,
    AmbientLight,
    DirectionalLight,
    Mesh,
    MeshStandardMaterial,
    TransformGizmo,
    WgpuRenderer,
    sphere_geometry,
    load_mesh,
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
    create_canvas,
)


# TODO:
# - better import paths
# - docstrings and typing
# - test wireframe
# - gizmo test
# - test drop shadow
# - ECS gui


canvas = create_canvas(max_fps=60, title="Hello ECS")
renderer = WgpuRenderer(canvas)


def setup(ecs: ECS, _):
    camera = PerspectiveCamera(70, 16 / 9)
    camera.local.position = (2, 2, 2)
    camera.look_at((0, 0, 0))

    ecs.spawn([SceneObject(camera), ActiveCamera()])

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
    gizmo.add_default_event_handlers(renderer, camera)

    ecs.spawn([SceneObject(gizmo, layer=1)])

    ecs.spawn([SceneObject(Background.from_color((0.9, 0.9, 0.9)))])

    ecs.spawn([SceneObject(AmbientLight(intensity=0.6))])
    ecs.spawn([SceneObject(DirectionalLight())])

    ecs.spawn([SceneObject(GridHelper())])

    controller = OrbitController(ecs, camera, target=(0, 0, 0))
    controller.add_default_event_handlers(renderer)


(
    ECS()
    .on("setup", setup)
    .do(performance_monitor)
    .do(ecs_explorer)
    .do(render_system, renderer)
    .do(render_gui_system)
    .do(window_system, canvas, "Hello ECS")
)
