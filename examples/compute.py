from wgpu.utils.compute import compute_with_buffers
import numpy as np
import time
import wgpu


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
shader_source = """
@group(0) @binding(0)
var<storage,read> data1: array<i32>;

@group(0) @binding(1)
var<storage,read_write> data2: array<i32>;

@compute
@workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {
    let i: u32 = index.x;
    data2[i] = data1[i] * data1[i];
}
"""
shader_source = shader_source.replace("WORKGROUP_SIZE", str(workgroup_size))

out = {}
with Timer("compute_with_buffer", 100) as rep:
    for _ in range(rep):
        out = compute_with_buffers(
            {0: numpy_data},
            {1: numpy_data.nbytes},
            shader_source,
            n=n // workgroup_size,
        )


with Timer("Building np.array from memoryview"):
    result = np.frombuffer(out[1], dtype=np.int32)

with Timer("Building a list from memoryview"):
    result = out[1].tolist()

# The long version using the wgpu API

# Create device and shader object
device = wgpu.utils.get_default_device()

with Timer("Create Pipeline"):
    cshader = device.create_shader_module(code=shader_source)

    # Setup layout and bindings
    binding_layouts = [
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage,
            },
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
    ]

    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    # Create and run the pipeline
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cshader, "entry_point": "main"},
    )

# Create buffer objects, input buffer is mapped.
with Timer("Create Buffers and bind group"):
    buffer1 = device.create_buffer_with_data(
        data=numpy_data, usage=wgpu.BufferUsage.STORAGE
    )
    buffer2 = device.create_buffer(
        size=numpy_data.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    bindings = [
        {
            "binding": 0,
            "resource": {"buffer": buffer1, "offset": 0, "size": buffer1.size},
        },
        {
            "binding": 1,
            "resource": {"buffer": buffer2, "offset": 0, "size": buffer2.size},
        },
    ]

    # Put everything together
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)


with Timer("dispatch", 1000) as rep:
    for _ in range(rep):
        command_encoder = device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(compute_pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(n // workgroup_size, 1, 1)  # x y z
        compute_pass.end()
        device.queue.submit([command_encoder.finish()])

    out = device.queue.read_buffer(buffer2).cast("i")
