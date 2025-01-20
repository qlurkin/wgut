import os
import re
from wgpu import GPUBindGroup, GPUBindGroupLayout
from wgut.builders import (
    CommandBufferBuilder,
    ComputePipelineBuilder,
    PipelineLayoutBuilder,
    compile_slang,
)


wg_pattern = re.compile(r"@workgroup_size\(([^)]+)\)")


class Computer:
    def __init__(
        self,
        source_or_filepath: str,
        bind_group_layouts: list[GPUBindGroupLayout],
        replace: dict[str, str] | None = None,
    ):
        if replace is None:
            replace = {}

        source = source_or_filepath
        if "\n" not in source_or_filepath:  # probably a filepath
            if os.path.exists(source_or_filepath):
                if source_or_filepath.endswith(".slang"):
                    source = compile_slang(source_or_filepath)
                else:
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

        pipeline_layout = (
            PipelineLayoutBuilder().with_bind_group_layout(bind_group_layouts).build()
        )

        self.source = source
        self.pipeline = (
            ComputePipelineBuilder()
            .with_shader_source(source)
            .with_layout(pipeline_layout)
            .build()
        )

    def dispatch(self, bind_groups: list[GPUBindGroup], x: int, y: int = 1, z: int = 1):
        command_encoder = CommandBufferBuilder()

        compute_pass = command_encoder.begin_compute_pass().build()

        compute_pass.set_pipeline(self.pipeline)
        for i, group in enumerate(bind_groups):
            compute_pass.set_bind_group(i, group)
        compute_pass.dispatch_workgroups(x, y, z)
        compute_pass.end()

        command_encoder.submit()
