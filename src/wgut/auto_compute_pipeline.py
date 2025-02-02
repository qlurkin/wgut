from wgut.reflection import Reflection
from wgut.builders import (
    BufferBuilder,
    compile_slang,
    BindGroupBuilder,
    CommandBufferBuilder,
    ComputePipelineBuilder,
)
import numpy.typing as npt
from wgpu import GPUBuffer


class AutoComputePipeline:
    def __init__(self, shader_source_file: str):
        if shader_source_file.endswith(".slang"):
            self.wgsl_source = compile_slang(shader_source_file)
        else:
            with open(shader_source_file) as file:
                self.wgsl_source = file.read()

        self.reflection = Reflection(self.wgsl_source)
        self.bind_groups = {}
        self.bindings = {}
        for gid in self.reflection.get_bind_group_ids():
            self.bind_groups[gid] = None
            self.bindings[gid] = {}
            for bid in self.reflection.get_binding_ids(gid):
                self.bindings[gid][bid] = None
        self.pipeline = (
            ComputePipelineBuilder()
            .with_layout(self.reflection.get_pipeline_layout())
            .with_shader_source(self.wgsl_source)
            .build()
        )

    def set_binding(
        self, group: int, binding: int, buffer: npt.NDArray | GPUBuffer
    ) -> GPUBuffer:
        if not isinstance(buffer, GPUBuffer):
            buffer = BufferBuilder().from_data(buffer).build()
        self.bind_groups[group] = None
        self.bindings[group][binding] = buffer
        return buffer

    def dispatch(self, x: int, y: int = 1, z: int = 1):
        for gid in self.bind_groups:
            if self.bind_groups[gid] is None:
                layout = self.reflection.get_bind_group_layout(gid)
                builder = BindGroupBuilder(layout)
                for bid in self.bindings[gid]:
                    builder.with_buffer(self.bindings[gid][bid], index=bid)
                self.bind_groups[gid] = builder.build()
        command_encoder = CommandBufferBuilder()
        compute_pass = command_encoder.begin_compute_pass().build()
        compute_pass.set_pipeline(self.pipeline)
        for gid, g in self.bind_groups.items():
            compute_pass.set_bind_group(gid, g)
        compute_pass.dispatch_workgroups(x, y, z)
        compute_pass.end()

        command_encoder.submit()
