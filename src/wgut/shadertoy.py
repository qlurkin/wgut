from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiRenderer

from wgut.builders import (
    BindGroupBuilder,
    BingGroupLayoutBuilder,
    BufferBuilder,
    CommandBufferBuilder,
    PipelineLayoutBuilder,
    RenderPipelineBuilder,
)
from wgut.core import (
    write_buffer,
    get_device,
)
from wgut.window import Window
import wgpu
import numpy as np
from datetime import datetime
from typing import Self

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
    def with_shader_source(self, source: str) -> Self:
        super().__init__()
        self.shader = shader_header + source + shader_footer
        return self

    def with_shader(self, filename: str) -> Self:
        with open(filename) as file:
            source = file.read()

        self.with_shader_source(source)
        return self

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
        return np.array([resolution[0], resolution[1], 0.0], dtype=np.float32)

    def setup(self):
        self.set_title("ShaderToy")
        self.time = 0.0
        self.last_mouse_down = (0, 0)
        self.last_mouse_click = (0, 0)
        self.is_mouse_down = False
        self.is_mouse_up_in_this_frame = False

        self.iResolutionBuffer = (
            BufferBuilder()
            .from_data(self.getIResolution())
            .with_usage(wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.UNIFORM)
            .build()
        )
        self.iTimeBuffer = (
            BufferBuilder()
            .from_data(self.getITime())
            .with_usage(wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.UNIFORM)
            .build()
        )
        self.iTimeDeltaBuffer = (
            BufferBuilder()
            .from_data(self.getITimeDelta(0.0))
            .with_usage(wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.UNIFORM)
            .build()
        )
        self.iDateBuffer = (
            BufferBuilder()
            .from_data(self.getIDate())
            .with_usage(wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.UNIFORM)
            .build()
        )
        self.iMouseBuffer = (
            BufferBuilder()
            .from_data(self.getIMouse())
            .with_usage(wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.UNIFORM)
            .build()
        )

        bg_layout = (
            BingGroupLayoutBuilder()
            .with_buffer(wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT)
            .with_buffer(wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT)
            .with_buffer(wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT)
            .with_buffer(wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT)
            .with_buffer(wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT)
            .build()
        )

        p_layout = PipelineLayoutBuilder().with_bind_group_layout(bg_layout).build()

        self.pipeline = (
            RenderPipelineBuilder(self.get_texture_format())
            .with_layout(p_layout)
            .with_shader_source(self.shader)
            .build()
        )

        self.bind_group = (
            BindGroupBuilder(bg_layout)
            .with_buffer(self.iResolutionBuffer)
            .with_buffer(self.iTimeBuffer)
            .with_buffer(self.iTimeDeltaBuffer)
            .with_buffer(self.iDateBuffer)
            .with_buffer(self.iMouseBuffer)
            .build()
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

    def render(self, screen: wgpu.GPUTexture):
        command_encoder = CommandBufferBuilder()

        render_pass = command_encoder.begin_render_pass(screen).build()

        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group)
        render_pass.draw(6)
        render_pass.end()

        command_encoder.submit()

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
