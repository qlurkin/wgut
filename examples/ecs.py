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
    OrbitController,
)

from wgut import (
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
# - docstrings and typing
# - test wireframe
# - test drop shadow


canvas = create_canvas(max_fps=60, title="Hello ECS")
renderer = WgpuRenderer(canvas)


def setup(ecs: ECS, _):
    # Camera
    camera = PerspectiveCamera(70, 16 / 9)
    camera.local.position = (2, 2, 2)
    camera.look_at((0, 0, 0))
    controller = OrbitController(camera, target=(0, 0, 0))
    controller.register_events(renderer)
    ecs.spawn([SceneObject(camera), ActiveCamera()])

    # Ball
    def test_event(event):
        print("Ball clicked !!")

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
    ball.add_event_handler(test_event, "click")

    # Bunny
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

    # Background
    ecs.spawn([SceneObject(Background.from_color((0.9, 0.9, 0.9)))])

    # Lights
    ecs.spawn([SceneObject(AmbientLight(intensity=0.6))])
    ecs.spawn([SceneObject(DirectionalLight())])

    # Grid
    ecs.spawn([SceneObject(GridHelper())])


(
    ECS()
    .on("setup", setup)
    .do(performance_monitor)
    .do(ecs_explorer)
    .do(render_system, renderer)
    .do(render_gui_system)
    .do(window_system, canvas, "Hello ECS")
)
