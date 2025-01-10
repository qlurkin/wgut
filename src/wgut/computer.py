from wgpu import GPUBuffer
from wgut.builders import (
    BindGroupBuilder,
    CommandBufferBuilder,
    ComputePipelineBuilder,
    get_adapter,
)


def pgcd(a, b):
    if b == 0:
        return a
    else:
        return pgcd(b, a % b)


def _ppcm(a, b):
    return abs(a * b) // pgcd(a, b)


def ppcm(lst):
    res = _ppcm(lst[0], lst[1])
    for x in lst[2:]:
        res = _ppcm(res, x)
    return res


class Computer:
    def __init__(self, source, strides: list[int]):
        self.strides = strides
        # self.workgroup_size = get_adapter().limits[
        #     "max-compute-invocations-per-workgroup"
        # ]
        self.workgroup_size = 1024
        # self.max_dispatch_size = get_adapter().limits[
        #     "max-compute-workgroups-per-dimension"
        # ]
        # self.min_offset_alignment = ppcm(
        #     [get_adapter().limits["min-storage-buffer-offset-alignment"]] + strides
        # )
        # self.dispatch_size = 8
        #
        # self.dispatch_total_alignement = self.min_offset_alignment // min(strides)
        # self.dispatch_size_alignement = (
        #     self.dispatch_total_alignement // self.workgroup_size
        # )
        # self.dispatch_size = (
        #     self.max_dispatch_size // self.dispatch_size_alignement
        # ) * self.dispatch_size_alignement

        self.dispatch_size = 65535

        self.pipeline = (
            ComputePipelineBuilder()
            .with_shader_source(
                source.replace("WORKGROUP_SIZE", str(self.workgroup_size))
            )
            .build()
        )

    def dispatch(self, bindings: list[GPUBuffer], n: int):
        nb_full_dispatch = n // (self.workgroup_size * self.dispatch_size)
        remaining = n % (self.workgroup_size * self.dispatch_size)
        extra_dispatch_size = remaining // self.workgroup_size
        if remaining % self.workgroup_size != 0:
            extra_dispatch_size += 1

        command_encoder = CommandBufferBuilder()

        def sub_dispatch(offset: int, dispatch_size: int):
            # print(
            #     f"offset: {offset}, size: {dispatch_size}, workgroup size: {self.workgroup_size}"
            # )

            bind_group_builder = BindGroupBuilder(
                self.pipeline.get_bind_group_layout(0)
            )

            for i, buffer in enumerate(bindings):
                bind_group_builder.with_buffer_binding(buffer, offset * self.strides[i])

            bind_group = bind_group_builder.build()

            compute_pass = command_encoder.begin_compute_pass().build()

            compute_pass.set_pipeline(self.pipeline)
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(dispatch_size, 1, 1)  # x y z
            compute_pass.end()

        for i in range(nb_full_dispatch):
            sub_dispatch(
                i * self.dispatch_size * self.workgroup_size, self.dispatch_size
            )

        sub_dispatch(
            nb_full_dispatch * self.dispatch_size * self.workgroup_size,
            extra_dispatch_size,
        )

        command_encoder.submit()
