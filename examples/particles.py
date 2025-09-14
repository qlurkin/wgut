from wgpu import (
    BufferBindingType,
    BufferUsage,
    GPUBindGroupLayout,
    GPUCommandEncoder,
    GPUComputePassEncoder,
    GPUComputePipeline,
    ShaderStage,
)

from wgut.cgmath import vec4
from wgut.core import get_device, load_file, read_buffer, submit_command, write_buffer
from wgut.orbit_camera import OrbitCamera
from wgut.scene.instance_mesh import InstanceMesh
from wgut.scene.performance_monitor import performance_monitor
from wgut.scene.render_gui_system import render_gui_system
from wgut.scene.render_system import (
    ActiveCamera,
    CameraComponent,
    MaterialComponent,
    MeshComponent,
    render_system,
)
from wgut.scene.renderer import Renderer, Material
from wgut.scene.window_system import window_system
from wgut.scene.ecs import ECS
from wgut.scene.primitives.icosphere import icosphere
from wgut.scene.transform import Transform
from wgut.scene.ambiant_light import AmbiantLight
import random
import numpy as np


def create_compute_pipeline() -> tuple[GPUComputePipeline, GPUBindGroupLayout]:
    shader_source = load_file("./particles.wgsl")

    shader_module = get_device().create_shader_module(code=shader_source)

    bind_group_layout = get_device().create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": ShaderStage.COMPUTE,
                "buffer": {"type": BufferBindingType.storage},
            },
            {
                "binding": 1,
                "visibility": ShaderStage.COMPUTE,
                "buffer": {"type": BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": ShaderStage.COMPUTE,
                "buffer": {"type": BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = get_device().create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    pipeline = get_device().create_compute_pipeline(
        layout=pipeline_layout,
        compute={
            "module": shader_module,
            "entry_point": "main",
        },
    )

    return pipeline, bind_group_layout


def setup(ecs: ECS):
    mesh = icosphere(2)

    pipeline, bind_group_layout = create_compute_pipeline()

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

    positions_array = np.array(positions)
    positions_buffer = get_device().create_buffer(
        size=positions_array.nbytes,
        usage=BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC,  # type: ignore
    )
    write_buffer(positions_buffer, positions_array)

    velocities_array = np.array(velocities)
    velocities_buffer = get_device().create_buffer(
        size=velocities_array.nbytes,
        usage=BufferUsage.STORAGE | BufferUsage.COPY_DST,  # type: ignore
    )
    write_buffer(velocities_buffer, velocities_array)

    dt_buffer = get_device().create_buffer(
        size=4,
        usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,  # type: ignore
    )

    bind_group = get_device().create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": positions_buffer,
                    "offset": 0,
                    "size": positions_buffer.size,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": velocities_buffer,
                    "offset": 0,
                    "size": velocities_buffer.size,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": dt_buffer,
                    "offset": 0,
                    "size": dt_buffer.size,
                },
            },
        ],
    )

    def update(ecs: ECS, delta_time: float):
        write_buffer(dt_buffer, np.array(delta_time, dtype=np.float32))

        command_encoder: GPUCommandEncoder = get_device().create_command_encoder()
        compute_pass: GPUComputePassEncoder = command_encoder.begin_compute_pass()

        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(index=0, bind_group=bind_group)
        compute_pass.dispatch_workgroups(1)
        compute_pass.end()

        submit_command(command_encoder)

    def get_translations():
        mem = read_buffer(positions_buffer)
        translations = np.frombuffer(mem.cast("f"), dtype=np.float32)
        translations = translations.reshape((translations.size // 4, 4))
        return translations

    particle_mesh = InstanceMesh(mesh, get_translations)

    ecs.on("update", update)

    material = Material((1.0, 0.0, 0.0, 1.0))

    ecs.spawn(
        [
            MaterialComponent(material),
            MeshComponent(particle_mesh),
            Transform().set_scale(0.2),
        ]
    )

    camera = OrbitCamera((6, 4, 5), (0, 0, 0), 45, 0.1, 1000)

    ecs.spawn([CameraComponent(camera), ActiveCamera()], label="Camera")
    ecs.spawn(AmbiantLight.create(color=(1.0, 1.0, 1.0)))


def process_event(ecs: ECS, event):
    cam: CameraComponent = ecs.query_one(CameraComponent)
    if isinstance(cam.camera, OrbitCamera):
        cam.camera.process_event(event)


renderer = Renderer(30000, 150000, 2, 1024, 32)


(
    ECS()
    .on("setup", setup)
    .on("window_event", process_event)
    .do(performance_monitor)
    .do(
        render_system,
        renderer,
    )
    .do(render_gui_system)
    .do(window_system, "Hello Particles")
)
