import numpy as np
import time
import wgpu
from wgut.builders import (
    BufferBuilder,
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
        print(f"{self.msg}:", (time.perf_counter() - self.start) / self.div)


n = 300_000_000

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
    with open("compute.wgsl") as file:
        source = file.read()
    computer = Computer(source, [4, 4])


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


with Timer("Computer dispatch", 100) as rep:
    for _ in range(rep):
        computer.dispatch([buffer1, buffer2], n)
    out = read_buffer(buffer2)

with Timer("Building np.array from memoryview"):
    result = np.frombuffer(out.cast("i"), dtype=np.int32)

with Timer("Building a list from memoryview"):
    result = out.cast("i").tolist()

print("Result is OK:", all(x == 9 for x in result))
