from imgui_bundle import imgui
from pyglm.glm import array, float32, scale, vec3, vec4
from wgpu import BufferUsage, GPUBuffer
from wgut.auto_compute_pipeline import AutoComputePipeline
from wgut.builders.bufferbuilder import BufferBuilder
from wgut.core import load_file, write_buffer
from wgut.orbit_camera import OrbitCamera
from wgut.scene.instance_mesh import InstanceMesh
from wgut.scene.particles import Particles
from wgut.scene.render_gui_system import render_gui_system
from wgut.scene.render_system import (
    ActiveCamera,
    CameraComponent,
    Layer,
    MaterialComponent,
    MeshComponent,
    RenderStat,
    render_system,
)
from wgut.scene.renderer import Renderer
from wgut.scene.window_system import window_system
from wgut.scene.ecs import ECS, EntityNotFound
from wgut.scene.basic_color_material import BasicColorMaterial
from wgut.scene.primitives.icosphere import icosphere
from wgut.scene.transform import Transform
import random

default_layer = Layer("default")


def setup(ecs: ECS):
    mesh = icosphere(2)

    computer = AutoComputePipeline(load_file("particles.wgsl"))

    positions = []
    for _ in range(128):
        positions.append(
            vec4(
                random.random() - 0.5, random.random() - 0.5, random.random() - 0.5, 0.0
            )
        )

    velocities = []
    for _ in range(128):
        velocities.append(
            vec4(
                random.random() - 0.5, random.random() - 0.5, random.random() - 0.5, 0.0
            )
        )

    velocities_buffer = (
        BufferBuilder()
        .from_data(array(velocities))
        .with_usage(BufferUsage.STORAGE)
        .build()
    )

    dt_buffer = (
        BufferBuilder()
        .with_size(4)
        .with_usage(BufferUsage.UNIFORM | BufferUsage.COPY_DST)
        .build()
    )

    def callback(ecs: ECS, buffer: GPUBuffer, delta_time: float):
        write_buffer(dt_buffer, array(float32(delta_time)))
        computer.set_buffer(0, 0, buffer)
        computer.set_buffer(0, 1, velocities_buffer)
        computer.set_buffer(0, 2, dt_buffer)
        computer.dispatch(1)

    particles = Particles(
        array(positions),
        callback,
    )

    particle_mesh = InstanceMesh(mesh, particles.get_translations)

    ecs.on("update", particles.update)

    material = BasicColorMaterial((1.0, 0.0, 0.0))

    ecs.spawn(
        [
            MaterialComponent(material),
            MeshComponent(particle_mesh),
            Transform(scale(vec3(0.2))),
            default_layer,
        ]
    )

    camera = OrbitCamera((6, 4, 5), (0, 0, 0), 45, 0.1, 1000)

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


renderer = Renderer(30000, 150000, 512, 2, (1024, 1024, 32), 256)


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
    .do(window_system, "Hello Particles")
)
