import os
import re
from wgpu import GPUBuffer
from wgut.builders import (
    BindGroupBuilder,
    CommandBufferBuilder,
    ComputePipelineBuilder,
)


wg_pattern = re.compile(r"@workgroup_size\(([^)]+)\)")


class Computer:
    def __init__(self, source_or_filepath: str, replace: dict[str, str] | None = None):
        if replace is None:
            replace = {}

        source = source_or_filepath
        if "\n" not in source_or_filepath:  # probably a filepath
            if os.path.exists(source_or_filepath):
                with open(source_or_filepath) as file:
                    source = file.read()

        for k, v in replace.items():
            source = source.replace(k, v)

        match = wg_pattern.search(source)
        if match is None:
            raise ValueError("Can't find WorkGroup Size in Source")
        dims = tuple(int(part.strip()) for part in match.group(1).split(","))
        while len(dims) < 3:
            dims = dims + (1,)
        self.workgroup_size = dims

        self.pipeline = ComputePipelineBuilder().with_shader_source(source).build()

    def dispatch(self, bindings: list[GPUBuffer], x: int, y: int = 1, z: int = 1):
        command_encoder = CommandBufferBuilder()

        bind_group_builder = BindGroupBuilder(self.pipeline.get_bind_group_layout(0))

        for i, buffer in enumerate(bindings):
            bind_group_builder.with_buffer_binding(buffer)

        bind_group = bind_group_builder.build()

        compute_pass = command_encoder.begin_compute_pass().build()

        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(x, y, z)
        compute_pass.end()

        command_encoder.submit()
