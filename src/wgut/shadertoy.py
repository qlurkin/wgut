from imgui_bundle import imgui
from numpy.typing import NDArray
from wgpu.gui.glfw import WgpuCanvas
from wgpu.utils.imgui import ImguiRenderer

from wgut.core import (
    submit_command,
    write_buffer,
    get_device,
)
from wgut.window import Window
import wgpu
import numpy as np
from datetime import datetime

shader_header = """
@group(0) @binding(0) var<uniform> i_resolution: vec3<f32>;
@group(0) @binding(1) var<uniform> i_time: f32;
@group(0) @binding(2) var<uniform> i_time_delta: f32;
@group(0)@binding(3) var<uniform> i_date: vec4<f32>;
@group(0) @binding(4) var<uniform> i_mouse: vec4<f32>;

struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) frag_coord: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
    );
    let index = i32(in.vertex_index);
    var out: VertexOutput;
    out.pos = vec4<f32>(positions[index], 0.0, 1.0);
    out.frag_coord = ((positions[index] + 1.0) / 2.0) * i_resolution.xy;
    return out;
}

"""

shader_footer = """
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return main_image(in.frag_coord);
}
"""


class ShaderToy(Window):
    def __init__(self, canvas: WgpuCanvas, source: str):
        super().__init__(canvas)
        self.shader = shader_header + source + shader_footer

    def getIDate(self):
        now = datetime.now()
        return np.array(
            [
                now.year,
                now.month,
                now.day,
                now.second
                + now.minute * 60
                + now.hour * 3600
                + now.microsecond * 0.000001,
            ],
            dtype=np.float32,
        )

    def getIMouse(self):
        return np.array(
            [
                self.last_mouse_down[0],
                self.last_mouse_down[1],
                self.last_mouse_click[0] * 1 if self.is_mouse_down else -1,
                self.last_mouse_click[1] * 1 if self.is_mouse_up_in_this_frame else -1,
            ],
            dtype=np.float32,
        )

    def getITime(self):
        return np.array([self.time], dtype=np.float32)

    def getITimeDelta(self, time_delta):
        return np.array([time_delta], dtype=np.float32)

    def getIResolution(self):
        resolution = self.get_canvas().get_physical_size()
        return np.array([resolution[0], resolution[1], 0.0], dtype=np.float32)  # type: ignore

    def create_buffer(self, data: NDArray):
        buffer = get_device().create_buffer(
            size=data.nbytes,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.UNIFORM,  # type: ignore
        )
        write_buffer(buffer, data)
        return buffer

    def setup(self):
        self.set_title("ShaderToy")
        self.time = 0.0
        self.last_mouse_down = (0, 0)
        self.last_mouse_click = (0, 0)
        self.is_mouse_down = False
        self.is_mouse_up_in_this_frame = False

        self.iResolutionBuffer = self.create_buffer(self.getIResolution())
        self.iTimeBuffer = self.create_buffer(self.getITime())
        self.iTimeDeltaBuffer = self.create_buffer(self.getITimeDelta(0.0))
        self.iDateBuffer = self.create_buffer(self.getIDate())
        self.iMouseBuffer = self.create_buffer(self.getIMouse())

        bg_layout = get_device().create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
            ]
        )

        p_layout = get_device().create_pipeline_layout(bind_group_layouts=[bg_layout])

        shader_module = get_device().create_shader_module(code=self.shader)

        self.pipeline = get_device().create_render_pipeline(
            layout=p_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.back,
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

        self.bind_group = get_device().create_bind_group(
            layout=bg_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.iResolutionBuffer,
                        "offset": 0,
                        "size": self.iResolutionBuffer.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.iTimeBuffer,
                        "offset": 0,
                        "size": self.iTimeBuffer.size,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self.iTimeDeltaBuffer,
                        "offset": 0,
                        "size": self.iTimeDeltaBuffer.size,
                    },
                },
                {
                    "binding": 3,
                    "resource": {
                        "buffer": self.iDateBuffer,
                        "offset": 0,
                        "size": self.iDateBuffer.size,
                    },
                },
                {
                    "binding": 4,
                    "resource": {
                        "buffer": self.iMouseBuffer,
                        "offset": 0,
                        "size": self.iMouseBuffer.size,
                    },
                },
            ],
        )

        self.imgui_renderer = ImguiRenderer(
            get_device(), self.get_canvas(), self.get_texture_format()
        )

        self.imgui_renderer.set_gui(self.gui)

        self.frame_time = 0

    def process_event(self, event):
        if event["event_type"] == "pointer_move":
            if self.is_mouse_down:
                self.last_mouse_click = (event["x"], event["y"])
        if event["event_type"] == "pointer_up":
            self.is_mouse_down = False
            self.is_mouse_up_in_this_frame = True
        if event["event_type"] == "pointer_down":
            self.is_mouse_down = True
            self.last_mouse_down = (event["x"], event["y"])
            self.last_mouse_click = (event["x"], event["y"])
        if event["event_type"] == "resize":
            write_buffer(self.iResolutionBuffer, self.getIResolution())

    def update(self, delta_time: float):
        self.frame_time = delta_time
        self.time += delta_time
        write_buffer(self.iTimeDeltaBuffer, self.getITimeDelta(delta_time))
        write_buffer(self.iTimeBuffer, self.getITime())
        write_buffer(self.iDateBuffer, self.getIDate())
        write_buffer(self.iMouseBuffer, self.getIMouse())

    def render(self):
        # command_encoder = CommandBufferBuilder()
        command_encoder = get_device().create_command_encoder()

        # render_pass = command_encoder.begin_render_pass(screen).build()
        render_pass: wgpu.GPURenderPassEncoder = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.get_current_texture().create_view(),
                    "resolve_target": None,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )

        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group)
        render_pass.draw(6)
        render_pass.end()

        submit_command(command_encoder)

        self.imgui_renderer.render()

        self.is_mouse_up_in_this_frame = False

    def gui(self) -> imgui.ImDrawData:
        imgui.new_frame()
        imgui.begin("Info", None)
        if self.frame_time != 0:
            imgui.text(f"FPS: {1.0 / self.frame_time:.5f}")
        else:
            imgui.text("FPS: NaN")
        imgui.text(f"i_delta_time: {self.frame_time:.5f}s")
        imgui.end()
        imgui.end_frame()
        imgui.render()
        return imgui.get_draw_data()
