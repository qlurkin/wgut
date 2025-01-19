from wgut.builders import (
    BindGroupBuilder,
    BufferBuilder,
    CommandBufferBuilder,
    write_buffer,
    GraphicPipelineBuilder,
)
from wgut.window import Window
import wgpu
import numpy as np
from datetime import datetime
from typing import Self

shader_header = """
@group(0) @binding(0) var<uniform> iResolution: vec3<f32>;
@group(0) @binding(1) var<uniform> iTime: f32;
@group(0) @binding(2) var<uniform> iTimeDelta: f32;
@group(0) @binding(3) var<uniform> iDate: vec4<f32>;
@group(0) @binding(4) var<uniform> iMouse: vec4<f32>;

struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) fragCoord: vec2<f32>,
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
    let time = iTime;
    let time_delta = iTimeDelta;
    let date = iDate;
    let mouse = iMouse;
    let index = i32(in.vertex_index);
    var out: VertexOutput;
    out.pos = vec4<f32>(positions[index], 0.0, 1.0);
    out.fragCoord = ((positions[index] + 1.0) / 2.0) * iResolution.xy;
    return out;
}

"""

shader_footer = """
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return mainImage(in.fragCoord);
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

        self.pipeline = (
            GraphicPipelineBuilder(self.get_texture_format())
            .with_shader_source(self.shader)
            .build()
        )

        self.bind_group = (
            BindGroupBuilder(self.pipeline.get_bind_group_layout(0))
            .with_buffer_binding(self.iResolutionBuffer)
            .with_buffer_binding(self.iTimeBuffer)
            .with_buffer_binding(self.iTimeDeltaBuffer)
            .with_buffer_binding(self.iDateBuffer)
            .with_buffer_binding(self.iMouseBuffer)
            .build()
        )

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
        self.is_mouse_up_in_this_frame = False
