import wgpu
from wgpu import TextureFormat, TextureUsage

TEXTURE_FORMAT = TextureFormat.rgba8unorm
ADAPTER = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
DEVICE = ADAPTER.request_device_sync()


class Texture:
    def __init__(self, size):
        self.size = size
        self.texture = DEVICE.create_texture(
            size=size,
            format=TEXTURE_FORMAT,
            usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC,
        )
        self.view = self.texture.create_view()

    def get_memoryview(self):
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


class GraphicPipeline:
    def __init__(self, shader_filename):
        with open(shader_filename) as file:
            shader_source = file.read()

        shader = DEVICE.create_shader_module(code=shader_source)

        # pipeline_layout = DEVICE.create_pipeline_layout(bind_group_layouts=[])

        self.render_pipeline = DEVICE.create_render_pipeline(
            layout="auto",
            vertex={
                "module": shader,
                "entry_point": "vs_main",
            },
            depth_stencil=None,
            multisample=None,
            fragment={
                "module": shader,
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
        )

    def render(self, texture):
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
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()
        DEVICE.queue.submit([command_encoder.finish()])
