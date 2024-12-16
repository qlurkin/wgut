import wgpu
from wgpu import TextureFormat, TextureUsage, VertexFormat, VertexStepMode
import numpy.typing as npt

TEXTURE_FORMAT = TextureFormat.rgba8unorm
ADAPTER = wgpu.gpu.request_adapter_sync(power_preference="high-performance")  # type: ignore
DEVICE = ADAPTER.request_device_sync()


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

    def shader(self, filename):
        with open(filename) as file:
            shader_source = file.read()
        self.shader_module = DEVICE.create_shader_module(code=shader_source)
        return self

    def build(self):
        # pipeline_layout = DEVICE.create_pipeline_layout(bind_group_layouts=[])
        params = {
            "layout": "auto",
            "vertex": {
                "module": self.shader_module,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": 4 * 5,
                        "step_mode": VertexStepMode.vertex,
                        "attributes": [
                            {
                                "format": VertexFormat.float32x2,
                                "offset": 0,
                                "shader_location": 0,
                            },
                            {
                                "format": VertexFormat.float32x3,
                                "offset": 4 * 2,
                                "shader_location": 1,
                            },
                        ],
                    },
                ],
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
