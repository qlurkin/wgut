from wgpu import BufferUsage, GPUBuffer
from wgut.auto_compute_pipeline import AutoComputePipeline
from wgut.builders.bufferbuilder import BufferBuilder
from wgut.cgmath import vec4
from wgut.core import load_file, write_buffer
from wgut.orbit_camera import OrbitCamera
from wgut.scene.instance_mesh import InstanceMesh
from wgut.scene.particles import Particles
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
from wgut.scene.basic_color_material import BasicColorMaterial
from wgut.scene.primitives.icosphere import icosphere
from wgut.scene.transform import Transform
import random
import numpy as np

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
        .from_data(np.array(velocities))
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
        write_buffer(dt_buffer, np.array(delta_time, dtype=np.float32))
        computer.set_buffer(0, 0, buffer)
        computer.set_buffer(0, 1, velocities_buffer)
        computer.set_buffer(0, 2, dt_buffer)
        computer.dispatch(1)

    particles = Particles(
        np.array(positions),
        callback,
    )

    particle_mesh = InstanceMesh(mesh, particles.get_translations)

    ecs.on("update", particles.update)

    material = BasicColorMaterial((1.0, 0.0, 0.0))

    ecs.spawn(
        [
            MaterialComponent(material),
            MeshComponent(particle_mesh),
            Transform().set_scale(0.2),
            default_layer,
        ]
    )

    camera = OrbitCamera((6, 4, 5), (0, 0, 0), 45, 0.1, 1000)

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
    .do(
        render_system,
        renderer,
        [default_layer],
    )
    .do(render_gui_system)
    .do(window_system, "Hello Particles")
)
