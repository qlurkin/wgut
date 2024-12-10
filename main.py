import wgpu
from wgpu.enums import TextureFormat
from wgpu.flags import TextureUsage
import pygame
import time

pygame.init()
pygame.font.init()

font = pygame.Font(None, 30)

size = (800, 600)
width, height = size


adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()


render_texture_format = TextureFormat.rgba8unorm

texture = device.create_texture(
    size=size,
    format=render_texture_format,
    usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC,
)

view = texture.create_view()


with open("shader.wgsl") as file:
    shader_source = file.read()

shader = device.create_shader_module(code=shader_source)

pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

render_pipeline = device.create_render_pipeline(
    layout=pipeline_layout,
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
                "format": render_texture_format,
                "blend": {
                    "color": {},
                    "alpha": {},
                },
            },
        ],
    },
)


def render():
    start = time.time()
    command_encoder = device.create_command_encoder()
    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            {
                "view": view,
                "resolve_target": None,
                "clear_value": (0, 0, 0, 1),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }
        ]
    )
    render_pass.set_pipeline(render_pipeline)
    render_pass.draw(3, 1, 0, 0)
    render_pass.end()
    device.queue.submit([command_encoder.finish()])

    buffer = device.queue.read_texture(
        source={
            "texture": texture,
            "origin": (0, 0, 0),
        },
        data_layout={
            "bytes_per_row": width * 4,
            "rows_per_image": height,
        },
        size=(width, height, 1),
    )

    surface = pygame.image.frombuffer(buffer.tobytes(), size, "RGBA")
    render_time = time.time() - start
    return surface, render_time


screen = pygame.display.set_mode(size)

clock = pygame.time.Clock()

# boucle jusqu'à ce qu'on quitte
while not pygame.event.peek(pygame.QUIT):
    clock.tick(60)
    surface, render_time = render()
    screen.blit(surface, (0, 0))
    screen.blit(
        font.render(f"Render Time: {render_time*1000:.5f} ms", False, (255, 255, 255))
    )
    pygame.display.flip()
