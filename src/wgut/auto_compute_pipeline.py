from wgut.reflection import Reflection
from wgut.builders import (
    BufferBuilder,
    BindGroupBuilder,
    CommandBufferBuilder,
    ComputePipelineBuilder,
)
import numpy.typing as npt
from wgpu import BufferUsage, GPUBuffer


class AutoComputePipeline:
    def __init__(self, shader_source: str):
        self.wgsl_source = shader_source
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

    def set_array(
        self,
        group: int,
        binding: int,
        array: npt.NDArray,
        additional_usages: BufferUsage | int | str = 0,
    ) -> GPUBuffer:
        builder = BufferBuilder().from_data(array)
        if self.reflection.get_binding_space(group, binding) == "uniform":
            builder.with_usage(BufferUsage.UNIFORM | additional_usages)  # type: ignore
        else:
            builder.with_usage(BufferUsage.STORAGE | additional_usages)  # type: ignore

        buffer = builder.build()
        self.set_buffer(group, binding, buffer)
        return buffer

    def set_buffer(self, group: int, binding: int, buffer: GPUBuffer) -> GPUBuffer:
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
