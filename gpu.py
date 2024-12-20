import wgpu
from wgpu import TextureFormat, TextureUsage, VertexFormat, VertexStepMode
import numpy.typing as npt

TEXTURE_FORMAT = TextureFormat.rgba8unorm
ADAPTER = wgpu.gpu.request_adapter_sync(power_preference="high-performance")  # type: ignore
DEVICE = ADAPTER.request_device_sync()

VERTEX_FORMAT_SIZE = {
    VertexFormat.float16x2: 4,
    VertexFormat.float16x4: 8,
    VertexFormat.float32: 4,
    VertexFormat.float32x2: 8,
    VertexFormat.float32x3: 12,
    VertexFormat.float32x4: 16,
    VertexFormat.sint16x2: 4,
    VertexFormat.sint16x4: 8,
    VertexFormat.sint32: 4,
    VertexFormat.sint32x2: 8,
    VertexFormat.sint32x3: 12,
    VertexFormat.sint32x4: 16,
    VertexFormat.sint8x2: 2,
    VertexFormat.sint8x4: 4,
    VertexFormat.snorm16x2: 4,
    VertexFormat.snorm16x4: 8,
    VertexFormat.snorm8x2: 2,
    VertexFormat.snorm8x4: 4,
    VertexFormat.uint16x2: 4,
    VertexFormat.uint16x4: 8,
    VertexFormat.uint32: 4,
    VertexFormat.uint32x2: 8,
    VertexFormat.uint32x3: 12,
    VertexFormat.uint32x4: 16,
    VertexFormat.uint8x2: 2,
    VertexFormat.uint8x4: 4,
    VertexFormat.unorm10_10_10_2: 4,
    VertexFormat.unorm16x2: 4,
    VertexFormat.unorm16x4: 8,
    VertexFormat.unorm8x2: 2,
    VertexFormat.unorm8x4: 4,
}


class Texture:
    def __init__(self, size: tuple[int, int]):
        self.size = size
        self.texture = DEVICE.create_texture(
            size=size,
            format=TEXTURE_FORMAT,
            usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC,
        )
        self.view = self.texture.create_view()

    def get_memoryview(self) -> memoryview:
        width, height = self.size
        buffer = DEVICE.queue.read_texture(
            source={
                "texture": self.texture,
                "origin": (0, 0, 0),
            },
            data_layout={
                "bytes_per_row": width * 4,
                "rows_per_image": height,
            },
            size=(width, height, 1),
        )
        return buffer


class Buffer:
    def __init__(self, data: npt.NDArray):
        self.data = DEVICE.create_buffer_with_data(
            data=data, usage=wgpu.BufferUsage.VERTEX
        )


class GraphicPipelineBuilder:
    def __init__(self):
        self.shader_module = None
        self.buffers = []
        self.location = 0

    def with_shader(self, filename):
        with open(filename) as file:
            shader_source = file.read()
        self.shader_module = DEVICE.create_shader_module(code=shader_source)
        return self

    def with_buffer(
        self, step_mode: VertexStepMode | str, array_stride: int | None = None
    ):
        self.buffers.append(
            {
                "array_stride": array_stride,
                "step_mode": step_mode,
                "attributes": [],
            }
        )
        return self

    def with_vertex_buffer(self, array_stride: int | None = None):
        self.with_buffer(VertexStepMode.vertex, array_stride)
        return self

    def with_instance_buffer(self, array_stride: int | None = None):
        self.with_buffer(VertexStepMode.instance, array_stride)
        return self

    def with_attribute(
        self,
        format: VertexFormat | str,
        offset: int | None = None,
        location: int | None = None,
    ):
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
        # pipeline_layout = DEVICE.create_pipeline_layout(bind_group_layouts=[])

        for buffer in self.buffers:
            if buffer["array_stride"] is None:
                last_attribute = buffer["attributes"][-1]
                buffer["array_stride"] = (
                    last_attribute["offset"]
                    + VERTEX_FORMAT_SIZE[last_attribute["format"]]
                )

        params = {
            "layout": "auto",
            "vertex": {
                "module": self.shader_module,
                "entry_point": "vs_main",
                "buffers": self.buffers,
            },
            "depth_stencil": None,
            "multisample": None,
            "fragment": {
                "module": self.shader_module,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": TEXTURE_FORMAT,
                        "blend": {
                            "color": {},
                            "alpha": {},
                        },
                    },
                ],
            },
        }
        return GraphicPipeline(params)


class GraphicPipeline:
    def __init__(self, params: dict):
        self.render_pipeline = DEVICE.create_render_pipeline(**params)
        # get_bind_group_layout(index: int)
        self.vertex_buffer = None

    def set_vertex_buffer(self, buffer: Buffer):
        self.vertex_buffer = buffer

    def render(self, texture: Texture):
        if self.vertex_buffer is None:
            return
        command_encoder = DEVICE.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": texture.view,
                    "resolve_target": None,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )
        render_pass.set_pipeline(self.render_pipeline)
        render_pass.set_vertex_buffer(0, self.vertex_buffer.data)
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()
        DEVICE.queue.submit([command_encoder.finish()])


class RenderPass:
    def __init__(self, texture: Texture):
        self.vertex_buffer_index = 0
        self.command_encoder = DEVICE.create_command_encoder()
        self.render_pass = self.command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": texture.view,
                    "resolve_target": None,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )

    def with_pipeline(self, pipeline: GraphicPipeline):
        self.render_pass.set_pipeline(pipeline.render_pipeline)
        return self

    def with_vertex_buffer(self, buffer: Buffer):
        self.render_pass.set_vertex_buffer(self.vertex_buffer_index, buffer.data)
        return self

    def draw(self, vertex_count, instance_count=1):
        self.render_pass.draw(vertex_count, instance_count, 0, 0)
        return self

    def submit(self):
        self.render_pass.end()
        DEVICE.queue.submit([self.command_encoder.finish()])
