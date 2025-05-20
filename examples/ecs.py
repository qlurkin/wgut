from wgut.orbit_camera import OrbitCamera
from wgut.scene.render_system import (
    ActiveCamera,
    CameraComponent,
    MaterialComponent,
    render_system,
)
from wgut.scene.window_system import window_system
from wgut.scene.ecs import ECS
from wgut.scene.pbr_material import PbrMaterial
from wgut.scene.primitives.icosphere import icosphere
from wgut.scene.transform import Transform


def setup(ecs: ECS):
    mesh = icosphere(3)
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
    ecs.spawn([mesh, MaterialComponent(material), transform])
    ecs.spawn([CameraComponent(camera), ActiveCamera()])


def process_event(ecs: ECS, event):
    cam = ecs.query_one([CameraComponent])
    cam[0].camera.process_event(event)


(
    ECS()
    .on("setup", setup)
    .on("render", render_system(10000, 50000, 128, (1024, 1024, 7), 48))
    .on("window_event", process_event)
    .do(window_system)
)
