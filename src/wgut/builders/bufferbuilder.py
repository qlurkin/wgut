from __future__ import annotations
from pyglm.glm import array
import wgpu

from .builderbase import BuilderBase
from ..core import get_device

import numpy.typing as npt
from typing import Self


class BufferBuilder(BuilderBase):
    def __init__(self):
        super().__init__()
        self.data = None
        self.size = None
        self.usages = None

    def from_data(self, data: npt.NDArray | array | bytes | memoryview) -> Self:
        data = memoryview(data)  # type: ignore
        self.with_size(data.nbytes)
        self.data = data
        return self

    def with_size(self, size: int) -> Self:
        if size < 0:
            raise Exception("size must be positive")
        if size % 4 != 0:
            raise Exception("size must be divisible by 4")
        self.size = size
        return self

    def with_usage(self, usage: wgpu.BufferUsage | int | str) -> Self:
        self.usages = usage
        return self

    def build(self) -> wgpu.GPUBuffer:
        if self.usages is None:
            raise Exception("Usage must be set")

        if self.data is not None:
            return get_device().create_buffer_with_data(
                data=self.data,
                usage=self.usages,  # type: ignore
            )
        if self.size is None:
            raise Exception("Size must be set")
        return get_device().create_buffer(
            label=self.label,
            size=self.size,
            usage=self.usages,  # type: ignore
            mapped_at_creation=False,
        )
