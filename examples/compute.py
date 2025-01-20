import numpy as np
import time
import wgpu
from wgut.builders import (
    BindGroupBuilder,
    BingGroupLayoutBuilder,
    BufferBuilder,
    get_adapter,
    read_buffer,
)
from wgut.computer import Computer


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


with Timer("Setup Computer"):
    bg_layout = (
        BingGroupLayoutBuilder()
        .with_buffer(wgpu.ShaderStage.COMPUTE, "read-only-storage")
        .with_buffer(wgpu.ShaderStage.COMPUTE, "storage")
        .build()
    )
    computer = Computer(
        "compute.wgsl",
        bind_group_layouts=[bg_layout],
        replace={
            "WORKGROUP_SIZE": str(workgroup_size),
            "Y_STRIDE": str(workgroup_size * dispatch_size) + "u",
        },
    )


with Timer("Create Buffers"):
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
    bg = BindGroupBuilder(bg_layout).with_buffer(buffer1).with_buffer(buffer2).build()


with Timer("Computer dispatch", 1000) as rep:
    for _ in range(rep):
        computer.dispatch([bg], n // (workgroup_size * mult), mult, 1)
    out = read_buffer(buffer2)

with Timer("Building np.array from memoryview"):
    result = np.frombuffer(out.cast("i"), dtype=np.int32)

with Timer("Building a list from memoryview"):
    result = out.cast("i").tolist()

print("Result is OK:", all(x == 9 for x in result) and len(result) == n)
