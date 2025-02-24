from __future__ import annotations
import wgpu

from .builderbase import BuilderBase
from ..core import get_device

from typing import Self


class SamplerBuilder(BuilderBase):
    def __init__(self):
        self.address_mode_u = wgpu.AddressMode.repeat
        self.address_mode_v = wgpu.AddressMode.repeat
        self.address_mode_w = wgpu.AddressMode.repeat
        self.mag_filter = wgpu.FilterMode.nearest
        self.min_filter = wgpu.FilterMode.nearest
        self.mipmap_filter = wgpu.MipmapFilterMode.nearest

    def with_address_mode_u(self, address_mode: wgpu.AddressMode | str) -> Self:
        self.address_mode_u = address_mode
        return self

    def with_address_mode_v(self, address_mode: wgpu.AddressMode | str) -> Self:
        self.address_mode_v = address_mode
        return self

    def with_address_mode_w(self, address_mode: wgpu.AddressMode | str) -> Self:
        self.address_mode_w = address_mode
        return self

    def with_address_mode(self, address_mode: wgpu.AddressMode | str) -> Self:
        self.address_mode_u = address_mode
        self.address_mode_v = address_mode
        self.address_mode_w = address_mode
        return self

    def with_mipmap_filter(self, mipmap_filter: wgpu.MipmapFilterMode | str) -> Self:
        self.mipmap_filter = mipmap_filter
        return self

    def with_mag_filter(self, filter: wgpu.FilterMode | str) -> Self:
        self.mag_filter = filter
        return self

    def with_min_filter(self, filter: wgpu.FilterMode | str) -> Self:
        self.min_filter = filter
        return self

    def with_filter(self, filter: wgpu.FilterMode | str) -> Self:
        self.mag_filter = filter
        self.min_filter = filter
        return self

    def build(self) -> wgpu.GPUSampler:
        return get_device().create_sampler(
            address_mode_u=self.address_mode_u,  # type: ignore
            address_mode_v=self.address_mode_v,  # type: ignore
            address_mode_w=self.address_mode_w,  # type: ignore
            mag_filter=self.mag_filter,  # type: ignore
            min_filter=self.min_filter,  # type: ignore
            mipmap_filter=self.mipmap_filter,  # type: ignore
        )
