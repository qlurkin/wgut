from __future__ import annotations
import wgpu

import numpy as np
import numpy.typing as npt
from typing import Self
from PIL import Image
import inspect
import subprocess as sp

from wgpu.enums import TextureFormat

_ADAPTER = None
_DEVICE = None

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


def get_adapter() -> wgpu.GPUAdapter:
    global _ADAPTER
    if _ADAPTER is None:
        _ADAPTER = wgpu.gpu.request_adapter_sync(power_preference="high-performance")  # type: ignore
    return _ADAPTER


def print_adapter_info():
    adapter = get_adapter()

    def title(text):
        return "\n" + text + ":\n" + "-" * (len(text) + 1)

    print(title("INFO"))
    for key, value in adapter.info.items():
        print(f" - {key}: {value}")
    print(f" - is_software: {adapter.is_fallback_adapter}")
    print(title("LIMITS"))
    for key, value in adapter.limits.items():
        print(f" - {key}: {value}")
    print(title("FEATURES"))
    for item in adapter.features:
        print(f" - {item}")
    print()


def get_device() -> wgpu.GPUDevice:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = get_adapter().request_device_sync()
    return _DEVICE


def compile_slang(filename: str) -> str:
    res = sp.run(["slangc", filename, "-target", "wgsl"], capture_output=True)
    if res.returncode != 0 or len(res.stderr) > 0:
        raise Exception(f"Slang compilation error: {res.stderr}")
    source = res.stdout
    return source.decode(encoding="utf8")


class BuilderBase:
    def __init__(self):
        name = type(self).__name__
        stack = inspect.stack()
        module_name = "wgut."
        module_path = "some_path"
        line = -1
        depth = 0
        while module_name.startswith("wgut."):
            parent_frame = stack[depth][0]
            module_info = inspect.getmodule(parent_frame)
            if module_info is None:
                break
            module_name = module_info.__name__
            module_path = module_info.__file__
            line = parent_frame.f_lineno
            depth += 1
        self.label = f"Made by {name} in {module_path} at line {line}"

    def with_label(self, label):
        self.label = label
        return self


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


class BufferBuilder(BuilderBase):
    def __init__(self):
        super().__init__()
        self.data = None
        self.size = None
        self.usages = None

    def from_data(self, data: npt.NDArray) -> Self:
        self.with_size(data.size * data.itemsize)
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


def read_buffer(buffer: wgpu.GPUBuffer) -> memoryview:
    return get_device().queue.read_buffer(buffer)


def write_buffer(buffer: wgpu.GPUBuffer, data: npt.NDArray):
    return get_device().queue.write_buffer(buffer=buffer, data=data, buffer_offset=0)


class PipelineBuilderBase(BuilderBase):
    def __init__(self):
        super().__init__()
        self.shader_module = None
        self.shader_source = ""
        self.layout = "auto"

    def with_layout(self, layout: wgpu.GPUPipelineLayout) -> Self:
        self.layout = layout
        return self

    def with_shader(self, filename: str, replace: None | dict[str, str] = None) -> Self:
        if replace is None:
            replace = {}
        with open(filename) as file:
            shader_source = file.read()
        for k, v in replace.items():
            shader_source = shader_source.replace(k, v)
        return self.with_shader_source(shader_source)

    def with_shader_source(self, source: str) -> Self:
        self.shader_source = source
        self.shader_module = get_device().create_shader_module(code=source)
        return self

    def with_slangc(self, filename) -> Self:
        source = compile_slang(filename)
        self.with_shader_source(source)
        return self


class BingGroupLayoutBuilder(BuilderBase):
    def __init__(self):
        super().__init__()
        self.entries = []
        self.index = 0

    def with_buffer(
        self,
        visibility: wgpu.ShaderStage | int,
        type: wgpu.BufferBindingType | str = "uniform",
        index: int | None = None,
    ) -> Self:
        if index is None:
            index = self.index
        self.entries.append(
            {
                "binding": index,
                "visibility": visibility,
                "buffer": {"type": type},
            }
        )
        self.index = index + 1
        return self

    def with_sampler(
        self,
        visibility: wgpu.ShaderStage | int,
        index: int | None = None,
    ) -> Self:
        if index is None:
            index = self.index
        self.entries.append(
            {
                "binding": index,
                "visibility": visibility,
                "sampler": {},
            }
        )
        self.index = index + 1
        return self

    def with_texture(
        self,
        visibility: wgpu.ShaderStage | int,
        index: int | None = None,
    ) -> Self:
        if index is None:
            index = self.index
        self.entries.append(
            {
                "binding": index,
                "visibility": visibility,
                "texture": {},
            }
        )
        self.index = index + 1
        return self

    def build(self) -> wgpu.GPUBindGroupLayout:
        return get_device().create_bind_group_layout(
            label=self.label, entries=self.entries
        )


class PipelineLayoutBuilder(BuilderBase):
    def __init__(self):
        super().__init__()
        self.layouts = []

    def with_bind_group_layout(
        self, layouts: wgpu.GPUBindGroupLayout | list[wgpu.GPUBindGroupLayout]
    ) -> Self:
        if not isinstance(layouts, list):
            layouts = [layouts]
        self.layouts += layouts
        return self

    def build(self) -> wgpu.GPUPipelineLayout:
        return get_device().create_pipeline_layout(
            label=self.label, bind_group_layouts=self.layouts
        )


class VertexBufferDescriptorsBuilder:
    def __init__(self):
        self.buffers = []
        self.location = 0

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
        return self.buffers


class RenderPipelineBuilder(PipelineBuilderBase):
    def __init__(
        self, output_format: TextureFormat | str, vertex_buffer_descriptors: list
    ):
        super().__init__()
        self.buffers = vertex_buffer_descriptors
        self.location = 0
        self.output_format = output_format
        self.depth_stencil_state = None

    def with_depth_stencil(
        self,
        depth_format: wgpu.TextureFormat = wgpu.TextureFormat.depth32float,  # type: ignore
        depth_write_enabled: bool = True,
        depth_compare: wgpu.CompareFunction = wgpu.CompareFunction.less,  # type: ignore
    ) -> Self:
        self.depth_stencil_state = {
            "format": depth_format,
            "depth_write_enabled": depth_write_enabled,
            "depth_compare": depth_compare,
        }
        return self

    def build(self) -> wgpu.GPURenderPipeline:
        # pipeline_layout = DEVICE.create_pipeline_layout(bind_group_layouts=[])

        for buffer in self.buffers:
            if buffer["array_stride"] is None:
                last_attribute = buffer["attributes"][-1]
                buffer["array_stride"] = (
                    last_attribute["offset"]
                    + VERTEX_FORMAT_SIZE[last_attribute["format"]]
                )

        return get_device().create_render_pipeline(
            label=self.label,
            layout=self.layout,  # type: ignore
            vertex={
                "module": self.shader_module,
                "entry_point": "vs_main",
                "buffers": self.buffers,
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.back,
            },
            depth_stencil=self.depth_stencil_state,
            multisample=None,
            fragment={
                "module": self.shader_module,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": self.output_format,
                        "blend": {
                            "color": {},
                            "alpha": {},
                        },
                    },
                ],
            },
        )


class ComputePipelineBuilder(PipelineBuilderBase):
    def __init__(self):
        super().__init__()

    def build(self) -> wgpu.GPURenderPipeline:
        return get_device().create_compute_pipeline(
            label=self.label,
            layout=self.layout,  # type: ignore
            compute={"module": self.shader_module, "entry_point": "main"},
        )


class CommandBufferBuilder(BuilderBase):
    def __init__(self):
        super().__init__()
        self.command_encoder = get_device().create_command_encoder(
            label=self.label,
        )

    def begin_render_pass(self, texture: wgpu.GPUTexture) -> RenderPassBuilder:
        return RenderPassBuilder(texture, self)

    def begin_compute_pass(self) -> ComputePassBuilder:
        return ComputePassBuilder(self)

    def build(self) -> Self:
        """NOOP -> Call Submit()"""
        return self

    def submit(self):
        get_device().queue.submit([self.command_encoder.finish()])


class RenderPassBuilder(BuilderBase):
    def __init__(
        self, texture: wgpu.GPUTexture, command_buffer_builder: CommandBufferBuilder
    ):
        super().__init__()
        self.command_buffer_builder = command_buffer_builder
        self.texture = texture
        self.clear_value = (0.9, 0.9, 0.9, 1)
        self.load_op = wgpu.LoadOp.clear
        self.store_op = wgpu.StoreOp.store
        self.depth_stencil_attachment = None

    def with_clear_value(self, value: tuple) -> Self:
        self.clear_value = value
        return self

    def with_load_op(self, load_op: wgpu.LoadOp | str) -> Self:
        self.load_op = load_op
        return self

    def with_store_op(self, store_op: wgpu.StoreOp | str) -> Self:
        self.store_op = store_op
        return self

    def with_depth_stencil(self, texture: wgpu.GPUTexture) -> Self:
        self.depth_stencil_attachment = {
            "view": texture.create_view(),
            "depth_clear_value": 1.0,
            "depth_load_op": wgpu.LoadOp.clear,
            "depth_store_op": wgpu.StoreOp.store,
        }
        return self

    def build(self) -> wgpu.GPURenderPassEncoder:
        return self.command_buffer_builder.command_encoder.begin_render_pass(
            label=self.label,
            color_attachments=[
                {
                    "view": self.texture.create_view(),
                    "resolve_target": None,
                    "clear_value": self.clear_value,
                    "load_op": self.load_op,
                    "store_op": self.store_op,
                }
            ],
            depth_stencil_attachment=self.depth_stencil_attachment,
        )


class ComputePassBuilder(BuilderBase):
    def __init__(self, command_buffer_builder: CommandBufferBuilder):
        super().__init__()
        self.command_buffer_builder = command_buffer_builder

    def build(self):
        return self.command_buffer_builder.command_encoder.begin_compute_pass(
            label=self.label,
        )


class BindGroupBuilder(BuilderBase):
    def __init__(self, layout):
        super().__init__()
        self.layout = layout
        self.bindings = []
        self.index = 0

    def with_buffer(
        self, buffer: wgpu.GPUBuffer, size=None, offset=0, index=None
    ) -> Self:
        if size is None:
            size = buffer.size - offset
        else:
            size = min(size, buffer.size - offset)

        if index is None:
            index = self.index

        self.bindings.append(
            {
                "binding": index,
                "resource": {
                    "buffer": buffer,
                    "offset": offset,
                    "size": size,
                },
            }
        )

        self.index = index + 1

        return self

    def with_texture(self, texture: wgpu.GPUTexture, index=None) -> Self:
        if index is None:
            index = self.index
        self.bindings.append({"binding": index, "resource": texture.create_view()})
        self.index = index + 1
        return self

    def with_sampler(
        self,
        address_mode: wgpu.AddressMode | str = wgpu.AddressMode.repeat,
        filter: wgpu.FilterMode | str = wgpu.FilterMode.nearest,
        index=None,
    ) -> Self:
        if index is None:
            index = self.index
        self.bindings.append(
            {
                "binding": index,
                "resource": get_device().create_sampler(
                    address_mode_u=address_mode,  # type: ignore
                    address_mode_v=address_mode,  # type: ignore
                    mag_filter=filter,  # type: ignore
                ),
            }
        )
        self.index = index + 1
        return self

    def build(self) -> wgpu.GPUBindGroup:
        return get_device().create_bind_group(
            label=self.label, layout=self.layout, entries=self.bindings
        )
