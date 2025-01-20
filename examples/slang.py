import numpy as np
import wgpu
from wgut.builders import (
    BindGroupBuilder,
    BufferBuilder,
    BingGroupLayoutBuilder,
    read_buffer,
)
from wgut.computer import Computer

bg_layout = (
    BingGroupLayoutBuilder()
    .with_buffer(wgpu.ShaderStage.COMPUTE, wgpu.BufferBindingType.read_only_storage)
    .with_buffer(wgpu.ShaderStage.COMPUTE, wgpu.BufferBindingType.read_only_storage)
    .with_buffer(wgpu.ShaderStage.COMPUTE, wgpu.BufferBindingType.storage)
    .build()
)

computer = Computer("./compute.slang", [bg_layout])

print(computer.source)

rng = np.random.default_rng()

numpy_data = rng.random(size=12800, dtype=np.float32)
print(numpy_data)
buffer1 = (
    BufferBuilder().from_data(numpy_data).with_usage(wgpu.BufferUsage.STORAGE).build()
)

numpy_data = rng.random(size=12800, dtype=np.float32)
print(numpy_data)
buffer2 = (
    BufferBuilder().from_data(numpy_data).with_usage(wgpu.BufferUsage.STORAGE).build()
)


buffer_res = (
    BufferBuilder()
    .with_size(numpy_data.nbytes)
    .with_usage(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    .build()
)

bg = (
    BindGroupBuilder(bg_layout)
    .with_buffer(buffer1)
    .with_buffer(buffer2)
    .with_buffer(buffer_res)
    .build()
)

computer.dispatch([bg], 12800 // 64)

out = read_buffer(buffer_res)

result = np.frombuffer(out.cast("f"), dtype=np.float32)

print(result)
