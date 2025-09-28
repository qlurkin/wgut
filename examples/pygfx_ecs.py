from pygfx import (
    PerspectiveCamera,
    Background,
    AmbientLight,
    DirectionalLight,
    Mesh,
    MeshStandardMaterial,
    sphere_geometry,
    load_mesh,
)

from wgut import (
    OrbitController,
    ActiveCamera,
    SceneObject,
    render_system,
    window_system,
    ECS,
    create_texture,
)


# TODO:
# - better import paths
# - docstrings and typing
# - wireframe
# - gizmo test
# - frustum culling
# - ply, gltf loaders
# - drop shadow
# - ECS gui


def setup(ecs: ECS, _):
    ball = Mesh(
        sphere_geometry(100, 100, 100),
        MeshStandardMaterial(
            map=create_texture("./textures/Wood_025_basecolor.jpg"),
            normal_map=create_texture("./textures/Wood_025_normal.jpg"),
            roughness_map=create_texture("./textures/Wood_025_roughness.jpg"),
        ),
    )
    ecs.spawn([SceneObject(ball)])
    ball.local.x -= 100

    bunny = load_mesh("./models/bunny.obj")[0]

    bunny.local.x += 100
    bunny.local.y -= 50
    bunny.local.scale = 1000

    bunny.material = MeshStandardMaterial(
        map=create_texture("./textures/texel_checker.png")
    )

    ecs.spawn([SceneObject(bunny)])

    ecs.spawn([SceneObject(Background.from_color((0.9, 0.9, 0.9)))])

    ecs.spawn([SceneObject(AmbientLight(intensity=0.6))])
    ecs.spawn([SceneObject(DirectionalLight())])

    camera = PerspectiveCamera(70, 16 / 9)
    camera.local.position = (200, 200, 200)
    camera.look_at((0, 0, 0))

    ecs.spawn([SceneObject(camera), ActiveCamera()])

    OrbitController(ecs, camera, target=(0, 0, 0))


(ECS().on("setup", setup).do(render_system).do(window_system, "Hello ECS"))
