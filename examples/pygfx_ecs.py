from wgut import (
    PerspectiveCamera,
    OrbitController,
    ActiveCamera,
    SceneObject,
    render_system,
    window_system,
    ECS,
    Background,
    AmbientLight,
    DirectionalLight,
    Mesh,
    box_geometry,
    MeshPhongMaterial,
    MeshPhysicalMaterial,
    MeshBasicMaterial,
    MeshStandardMaterial,
    icosahedron_geometry,
    sphere_geometry,
    torus_knot_geometry,
    Geometry,
)
from wgut.core import create_texture

MeshBasicMaterial()
Geometry()


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

    ecs.spawn([SceneObject(Background.from_color((0.9, 0.9, 0.9)))])

    ecs.spawn([SceneObject(AmbientLight(intensity=0.6))])
    ecs.spawn([SceneObject(DirectionalLight())])

    camera = PerspectiveCamera(70, 16 / 9)
    camera.local.position = (150, 150, 150)
    camera.look_at((0, 0, 0))

    ecs.spawn([SceneObject(camera), ActiveCamera()])

    OrbitController(ecs, camera, target=(0, 0, 0))


(ECS().on("setup", setup).do(render_system).do(window_system, "Hello ECS"))
