from wgpu import (
    BufferUsage,
)

from wgut import (
    load_file,
    performance_monitor,
    render_gui_system,
    render_system,
    create_canvas,
    ActiveCamera,
    SceneObject,
    window_system,
    ECS,
)

from pygfx import (
    Background,
    Buffer,
    InstancedMesh,
    OrbitController,
    PerspectiveCamera,
    icosahedron_geometry,
    AmbientLight,
    MeshBasicMaterial,
    WgpuRenderer,
)

from pygfx.utils.compute import ComputeShader

from pylinalg import mat_from_translation

import random
import numpy as np

canvas = create_canvas(max_fps=60, title="Hello Particles")
renderer = WgpuRenderer(canvas)


def setup(ecs: ECS):
    # particles
    particles_count = 128
    geometry = icosahedron_geometry(0.1, 3)
    material = MeshBasicMaterial(color=(1.0, 0, 0))
    particles = InstancedMesh(geometry, material, particles_count)

    for i in range(particles_count):
        m = mat_from_translation(
            (random.random() - 0.5, random.random() - 0.5, random.random() - 0.5),
        )
        particles.set_matrix_at(i, m)

    instance_buffer: Buffer = particles.instance_buffer
    instance_buffer._wgpu_usage = BufferUsage.COPY_DST | BufferUsage.STORAGE

    velocities = []
    for _ in range(particles_count):
        velocities.append(
            np.array(
                [
                    random.random() - 0.5,
                    random.random() - 0.5,
                    random.random() - 0.5,
                    0.0,
                ],
                dtype=np.float32,
            )
        )

    velocities_array = np.array(velocities)

    velocities_buffer = Buffer(
        velocities_array, usage=BufferUsage.COPY_DST | BufferUsage.STORAGE
    )

    dt_buffer = Buffer(
        np.array([0.0], dtype=np.float32),
        usage=BufferUsage.COPY_DST | BufferUsage.UNIFORM,
    )

    compute_shader = ComputeShader(load_file("./particles.wgsl"))
    compute_shader.set_resource(0, instance_buffer)
    compute_shader.set_resource(1, velocities_buffer)
    compute_shader.set_resource(2, dt_buffer)

    def update(ecs: ECS, delta_time: float):
        dt_buffer.set_data(np.array([delta_time], dtype=np.float32))
        dt_buffer.update_full()

        compute_shader.dispatch(1)

    ecs.on("update", update)

    ecs.spawn([SceneObject(particles)])

    # Background
    ecs.spawn([SceneObject(Background.from_color((0.9, 0.9, 0.9)))])

    # Camera
    camera = PerspectiveCamera(70, 16 / 9)
    camera.local.position = (2, 2, 2)
    camera.look_at((0, 0, 0))
    controller = OrbitController(camera, target=(0, 0, 0))
    controller.register_events(renderer)
    ecs.spawn([SceneObject(camera), ActiveCamera()])

    # Lights
    ecs.spawn([SceneObject(AmbientLight(intensity=0.6))])


(
    ECS()
    .on("setup", setup)
    .do(performance_monitor)
    .do(render_system, renderer)
    .do(render_gui_system, canvas)
    .do(window_system, canvas)
)
