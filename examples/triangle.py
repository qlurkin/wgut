from wgpu import (
    AutoLayoutMode,
    BufferUsage,
    CullMode,
    FrontFace,
    LoadOp,
    PrimitiveTopology,
    StoreOp,
    VertexFormat,
    VertexStepMode,
)
from wgut import (
    print_adapter_info,
    Window,
    load_file,
    get_device,
    submit_command,
    write_buffer,
    create_canvas,
)
import numpy as np

canvas = create_canvas()


class MyApp(Window):
    def setup(self):
        print_adapter_info()
        self.set_title("Hello Triangle")
        vertex_data = np.array(
            [
                # x, y, r, g, b
                [0.0, 0.5, 1.0, 0.0, 0.0],
                [-0.5, -0.5, 0.0, 1.0, 0.0],
                [0.5, -0.5, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.vertex_buffer = get_device().create_buffer(
            size=vertex_data.nbytes,
            usage=BufferUsage.VERTEX | BufferUsage.COPY_DST,  # type: ignore
        )
        write_buffer(self.vertex_buffer, vertex_data)

        vertex_buffer_descriptor = {
            "array_stride": 5 * 4,
            "step_mode": VertexStepMode.vertex,
            "attributes": [
                {
                    "format": VertexFormat.float32x2,
                    "offset": 0,
                    "shader_location": 0,
                },
                {
                    "format": VertexFormat.float32x3,
                    "offset": 2 * 4,
                    "shader_location": 1,
                },
            ],
        }

        shader_module = get_device().create_shader_module(
            code=load_file("triangle.wgsl")
        )

        self.pipeline = get_device().create_render_pipeline(
            label="Render Pipeline",
            layout=AutoLayoutMode.auto,  # type: ignore
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [vertex_buffer_descriptor],
            },
            primitive={
                "topology": PrimitiveTopology.triangle_list,
                "front_face": FrontFace.ccw,
                "cull_mode": CullMode.back,
            },
            depth_stencil=None,
            multisample=None,
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": self.get_texture_format(),
                        "blend": {
                            "color": {},
                            "alpha": {},
                        },
                    },
                ],
            },
        )

    def render(self):
        command_encoder = get_device().create_command_encoder()

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.get_current_texture().create_view(),
                    "resolve_target": None,
                    "clear_value": (0.9, 0.9, 0.9, 1.0),
                    "load_op": LoadOp.clear,
                    "store_op": StoreOp.store,
                }
            ],
        )

        render_pass.set_pipeline(self.pipeline)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(3)
        render_pass.end()

        submit_command(command_encoder)


MyApp(canvas).run()
