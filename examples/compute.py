import numpy as np
import time
import wgpu
from wgut import (
    BufferBuilder,
    get_adapter,
    read_buffer,
    AutoComputePipeline,
    load_file,
)


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
    computer = AutoComputePipeline(
        load_file("compute.wgsl")
        .replace("WORKGROUP_SIZE", str(workgroup_size))
        .replace("Y_STRIDE", str(workgroup_size * dispatch_size) + "u")
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
    computer.set_buffer(0, 0, buffer1)
    computer.set_buffer(0, 1, buffer2)


with Timer("Computer dispatch", 1000) as rep:
    for _ in range(rep):
        computer.dispatch(n // (workgroup_size * mult), mult)
    out = read_buffer(buffer2)

with Timer("Building np.array from memoryview"):
    result = np.frombuffer(out.cast("i"), dtype=np.int32)

with Timer("Building a list from memoryview"):
    result = out.cast("i").tolist()

print("Result is OK:", all(x == 9 for x in result) and len(result) == n)
