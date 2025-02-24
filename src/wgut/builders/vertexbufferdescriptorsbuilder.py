from __future__ import annotations
import wgpu

from typing import Self


VERTEX_FORMAT_SIZE = {
    wgpu.VertexFormat.float16x2: 4,
    wgpu.VertexFormat.float16x4: 8,
    wgpu.VertexFormat.float32: 4,
    wgpu.VertexFormat.float32x2: 8,
    wgpu.VertexFormat.float32x3: 12,
    wgpu.VertexFormat.float32x4: 16,
    wgpu.VertexFormat.sint16x2: 4,
    wgpu.VertexFormat.sint16x4: 8,
    wgpu.VertexFormat.sint32: 4,
    wgpu.VertexFormat.sint32x2: 8,
    wgpu.VertexFormat.sint32x3: 12,
    wgpu.VertexFormat.sint32x4: 16,
    wgpu.VertexFormat.sint8x2: 2,
    wgpu.VertexFormat.sint8x4: 4,
    wgpu.VertexFormat.snorm16x2: 4,
    wgpu.VertexFormat.snorm16x4: 8,
    wgpu.VertexFormat.snorm8x2: 2,
    wgpu.VertexFormat.snorm8x4: 4,
    wgpu.VertexFormat.uint16x2: 4,
    wgpu.VertexFormat.uint16x4: 8,
    wgpu.VertexFormat.uint32: 4,
    wgpu.VertexFormat.uint32x2: 8,
    wgpu.VertexFormat.uint32x3: 12,
    wgpu.VertexFormat.uint32x4: 16,
    wgpu.VertexFormat.uint8x2: 2,
    wgpu.VertexFormat.uint8x4: 4,
    wgpu.VertexFormat.unorm10_10_10_2: 4,
    wgpu.VertexFormat.unorm16x2: 4,
    wgpu.VertexFormat.unorm16x4: 8,
    wgpu.VertexFormat.unorm8x2: 2,
    wgpu.VertexFormat.unorm8x4: 4,
}


class VertexBufferDescriptorsBuilder:
    def __init__(self):
        self.buffers = []
        self.location = 0

    def with_vertex_descriptor(self, vertex_descriptor: dict) -> Self:
        max_location = self.location
        for attr in vertex_descriptor["attributes"]:
            attr["shader_location"] += self.location
            if attr["shader_location"] > max_location:
                max_location = attr["shader_location"]
        self.buffers.append(vertex_descriptor)
        self.location = max_location + 1
        return self

    def with_buffer(
        self, step_mode: wgpu.VertexStepMode | str, array_stride: int | None = None
    ) -> Self:
        self.buffers.append(
            {
                "array_stride": array_stride,
                "step_mode": step_mode,
                "attributes": [],
            }
        )
        return self

    def with_vertex_buffer(self, array_stride: int | None = None) -> Self:
        self.with_buffer(wgpu.VertexStepMode.vertex, array_stride)
        return self

    def with_instance_buffer(self, array_stride: int | None = None) -> Self:
        self.with_buffer(wgpu.VertexStepMode.instance, array_stride)
        return self

    def with_attribute(
        self,
        format: wgpu.VertexFormat | str,
        offset: int | None = None,
        location: int | None = None,
    ) -> Self:
        if location is None:
            location = self.location
        self.location = location + 1

        if offset is None:
            if len(self.buffers[-1]["attributes"]) == 0:
                offset = 0
            else:
                prev = self.buffers[-1]["attributes"][-1]
                offset = prev["offset"] + VERTEX_FORMAT_SIZE[prev["format"]]

        self.buffers[-1]["attributes"].append(
            {
                "format": format,
                "offset": offset,
                "shader_location": location,
            }
        )
        return self

    def build(self):
        for buffer in self.buffers:
            if buffer["array_stride"] is None:
                last_attribute = buffer["attributes"][-1]
                buffer["array_stride"] = (
                    last_attribute["offset"]
                    + VERTEX_FORMAT_SIZE[last_attribute["format"]]
                )
        return self.buffers
