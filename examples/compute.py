import numpy as np
import time
from wgpu import (
    BufferBindingType,
    BufferUsage,
    GPUCommandEncoder,
    GPUComputePassEncoder,
    ShaderStage,
)
from wgut import (
    get_adapter,
    read_buffer,
    load_file,
)
from wgut.core import get_device, submit_command, write_buffer


class Timer:
    def __init__(self, msg: str, div=1):
        self.msg = msg
        self.div = div

    def __enter__(self):
        self.start = time.perf_counter()
        return self.div

    def __exit__(self, exc_type, exc_value, traceback):
        avg_txt = ""
        if self.div > 1:
            avg_txt = f" (avg on {self.div} times)"
        print(
            f"{self.msg}{avg_txt}:",
            (time.perf_counter() - self.start) / self.div,
        )


workgroup_size = get_adapter().limits["max-compute-invocations-per-workgroup"]
dispatch_size = get_adapter().limits["max-compute-workgroups-per-dimension"]


mult = 1
n = workgroup_size * dispatch_size * mult

print("Array Size:", n)

with Timer("create numpy array"):
    numpy_data = np.full(n, 3, dtype=np.int32)


with Timer("for in range loop"):
    res = []
    for x in numpy_data:
        res.append(x * x)


with Timer("numpy operation", 100) as rep:
    for _ in range(rep):
        res = numpy_data * numpy_data

with Timer("Get Device"):
    get_device()

with Timer("Setup Computer"):
    shader_source = (
        load_file("compute.wgsl")
        .replace("WORKGROUP_SIZE", str(workgroup_size))
        .replace("Y_STRIDE", str(workgroup_size * dispatch_size) + "u")
    )

    shader_module = get_device().create_shader_module(code=shader_source)

    bind_group_layout = get_device().create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": ShaderStage.COMPUTE,
                "buffer": {"type": BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": ShaderStage.COMPUTE,
                "buffer": {"type": BufferBindingType.storage},
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


with Timer("Create Buffers"):
    buffer0 = get_device().create_buffer(
        size=numpy_data.nbytes,
        usage=BufferUsage.STORAGE | BufferUsage.COPY_DST,  # type: ignore
    )
    write_buffer(buffer0, numpy_data)

    buffer1 = get_device().create_buffer(
        size=numpy_data.nbytes,
        usage=BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC,  # type: ignore
    )

    bind_group = get_device().create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": buffer0,
                    "offset": 0,
                    "size": buffer0.size,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": buffer1,
                    "offset": 0,
                    "size": buffer1.size,
                },
            },
        ],
    )

with Timer("Computer dispatch", 1000) as rep:
    command_encoder: GPUCommandEncoder = get_device().create_command_encoder()

    compute_pass: GPUComputePassEncoder = command_encoder.begin_compute_pass()

    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(n // (workgroup_size * mult), mult)
    compute_pass.end()

    submit_command(command_encoder)
    out = read_buffer(buffer1)

with Timer("Building np.array from memoryview"):
    result = np.frombuffer(out.cast("i"), dtype=np.int32)

with Timer("Building a list from memoryview"):
    result = out.cast("i").tolist()

print("Result is OK:", all(x == 9 for x in result) and len(result) == n)
