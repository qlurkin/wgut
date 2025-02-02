import numpy as np
import wgpu
from wgut.auto_compute_pipeline import AutoComputePipeline
from wgut.builders import (
    BufferBuilder,
    read_buffer,
)


computer = AutoComputePipeline("./compute.slang")

rng = np.random.default_rng()

numpy_data = rng.random(size=12800, dtype=np.float32)
print(numpy_data)
buffer1 = (
    BufferBuilder().from_data(numpy_data).with_usage(wgpu.BufferUsage.STORAGE).build()
)

computer.set_binding(0, 0, buffer1)

numpy_data = rng.random(size=12800, dtype=np.float32)
print(numpy_data)
buffer2 = (
    BufferBuilder().from_data(numpy_data).with_usage(wgpu.BufferUsage.STORAGE).build()
)

computer.set_binding(0, 1, buffer2)


buffer_res = (
    BufferBuilder()
    .with_size(numpy_data.nbytes)
    .with_usage(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    .build()
)

computer.set_binding(0, 2, buffer_res)

computer.dispatch(12800 // 64)

out = read_buffer(buffer_res)

result = np.frombuffer(out.cast("f"), dtype=np.float32)

print(result)
