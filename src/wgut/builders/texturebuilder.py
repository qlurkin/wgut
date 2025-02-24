from __future__ import annotations
import wgpu

from ..core import get_device
from .builderbase import BuilderBase

import numpy as np
from typing import Self
from PIL import Image

from wgpu.enums import TextureFormat


class TextureBuilder(BuilderBase):
    def __init__(self):
        super().__init__()
        self.size = None
        self.format = None
        self.usages = None

    def with_size(self, size: tuple[int, int]) -> Self:
        self.size = size
        return self

    def with_format(self, format: wgpu.TextureFormat | str) -> Self:
        self.format = format
        return self

    def with_usage(self, usage: wgpu.TextureUsage | int | str) -> Self:
        self.usages = usage
        return self

    def from_file(self, filename: str, srgb: bool = True) -> wgpu.GPUTexture:
        image = Image.open(filename)
        data = np.asarray(image)
        if image.mode == "RGB":
            data = np.dstack((data, np.full(data.shape[:-1], 255, dtype=np.uint8)))
        format = TextureFormat.rgba8unorm
        if srgb:
            format = TextureFormat.rgba8unorm_srgb

        texture = (
            self.with_size(image.size)
            .with_format(format)
            .with_usage(wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING)
            .build()
        )

        get_device().queue.write_texture(
            {
                "texture": texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            data,
            {
                "offset": 0,
                "bytes_per_row": data.strides[0],
            },
            image.size,
        )

        return texture

    def build(self) -> wgpu.GPUTexture:
        if self.format is None:
            raise Exception("format must be set")

        if self.usages is None:
            raise Exception("Usages must be set")

        return get_device().create_texture(
            label=self.label,
            size=self.size,
            format=self.format,  # type: ignore
            usage=self.usages,  # type: ignore
        )

    def build_depth(self, size: tuple[int, int]) -> wgpu.GPUTexture:
        return (
            self.with_format(wgpu.TextureFormat.depth32float)
            .with_size(size)
            .with_usage(
                wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
            )
            .build()
        )
