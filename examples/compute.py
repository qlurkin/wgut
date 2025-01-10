import numpy as np
import time
import wgpu
from wgut.builders import (
    BindGroupBuilder,
    BufferBuilder,
    ComputePipelineBuilder,
    CommandBufferBuilder,
    read_buffer,
)


class Timer:
    def __init__(self, msg: str, div=1):
        self.msg = msg
        self.div = div

    def __enter__(self):
        self.start = time.perf_counter()
        return self.div

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"{self.msg}:", (time.perf_counter() - self.start) / self.div)


n = 25600000

with Timer("create numpy array"):
    numpy_data = np.full(n, 3, dtype=np.int32)


with Timer("for in range loop"):
    res = []
    for x in numpy_data:
        res.append(x * x)


with Timer("numpy operation", 1000) as rep:
    for _ in range(rep):
        res = numpy_data * numpy_data

workgroup_size = 512


# The long version using the wgpu API
with Timer("Create Pipeline"):
    compute_pipeline = (
        ComputePipelineBuilder()
        .with_shader("compute.wgsl", {"WORKGROUP_SIZE": str(workgroup_size)})
        .build()
    )

with Timer("Create Buffers and bind group"):
    buffer1 = (
        BufferBuilder()
        .from_data(numpy_data)
        .with_usage(wgpu.BufferUsage.STORAGE)
        .build()
    )
    buffer2 = (
        BufferBuilder()
        .with_size(numpy_data.nbytes)
        .with_usage(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        .build()
    )

    bind_group = (
        BindGroupBuilder(compute_pipeline.get_bind_group_layout(0))
        .with_buffer_binding(buffer1)
        .with_buffer_binding(buffer2)
        .build()
    )


with Timer("dispatch", 1000) as rep:
    for _ in range(rep):
        command_encoder = CommandBufferBuilder()

        compute_pass = command_encoder.begin_compute_pass().build()

        compute_pass.set_pipeline(compute_pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(n // workgroup_size, 1, 1)  # x y z
        compute_pass.end()

        command_encoder.submit()

    out = read_buffer(buffer2).cast("i")

with Timer("Building np.array from memoryview"):
    result = np.frombuffer(out, dtype=np.int32)

with Timer("Building a list from memoryview"):
    result = out.tolist()
