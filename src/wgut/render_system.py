from dataclasses import dataclass
from typing import Callable
from imgui_bundle import imgui
from pygfx import (
    Camera,
    WgpuRenderer,
    WindowEvent,
    WorldObject,
    Scene,
)
from wgut.ecs import ECS
from wgut.window import Window
from time import perf_counter


class ActiveCamera:
    def __str__(self):
        return "ActiveCamera"


@dataclass
class SceneObject:
    obj: WorldObject
    layer: int = 0

    def __str__(self):
        return str(self.obj.__class__.__name__)

    def ecs_explorer_gui(self):
        imgui.push_id("translation")
        if imgui.collapsing_header("translation"):
            changed, val = imgui.input_float("x", self.obj.local.x)
            if changed:
                self.obj.local.x = val
            changed, val = imgui.input_float("y", self.obj.local.y)
            if changed:
                self.obj.local.y = val
            changed, val = imgui.input_float("z", self.obj.local.z)
            if changed:
                self.obj.local.z = val
        imgui.pop_id()

        imgui.push_id("rotation")
        if imgui.collapsing_header("rotation (euler)"):
            changed, val = imgui.input_float("x", self.obj.local.euler_x)
            if changed:
                self.obj.local.euler_x = val
            changed, val = imgui.input_float("y", self.obj.local.euler_y)
            if changed:
                self.obj.local.euler_y = val
            changed, val = imgui.input_float("z", self.obj.local.euler_z)
            if changed:
                self.obj.local.euler_z = val
        imgui.pop_id()

        imgui.push_id("scale")
        if imgui.collapsing_header("scale"):
            changed, val = imgui.input_float("x", self.obj.local.scale_x)
            if changed:
                self.obj.local.scale_x = val
            changed, val = imgui.input_float("y", self.obj.local.scale_y)
            if changed:
                self.obj.local.scale_y = val
            changed, val = imgui.input_float("z", self.obj.local.scale_z)
            if changed:
                self.obj.local.scale_z = val
        imgui.pop_id()


def render_system(ecs: ECS, renderer: WgpuRenderer):
    def setup(ecs: ECS):
        scenes = {0: Scene()}
        stats = {}

        def clear():
            for scn in scenes.values():
                scn.clear()

        def dispatch(event):
            ecs.dispatch("pygfx_event", event)

        renderer.add_event_handler(
            dispatch,
            "resize",
            "close",
            "pointer_down",
            "pointer_up",
            "pointer_move",
            "pointer_enter",
            "pointer_leave",
            "double_click",
            "wheel",
            "key_down",
            "key_up",
            "before_render",
            "after_render",
        )

        def handle_stats(ecs: ECS, fn: Callable[[dict], None]):
            fn(stats)

        ecs.on("call_with_stats", handle_stats)

        def render(ecs: ECS):
            start = perf_counter()
            cam_so: SceneObject
            cam_so, _ = ecs.query_one([SceneObject, ActiveCamera])
            camera = cam_so.obj

            assert isinstance(camera, Camera), "The ActiveCamera is not a Camera"

            clear()
            for (so,) in ecs.query([SceneObject]):
                if so.layer not in scenes:
                    scenes[so.layer] = Scene()
                scenes[so.layer].add(so.obj)

            renderer.clear(all=True)
            for layer in sorted(scenes.keys()):
                renderer.render(scenes[layer], camera, flush=False)
                renderer.clear(depth=True)
            renderer.flush()
            renderer.dispatch_event(
                WindowEvent(
                    "after_render",
                    target=None,
                    root=renderer,
                    width=renderer.logical_size[0],
                    height=renderer.logical_size[1],  # type: ignore
                    pixel_ratio=renderer.pixel_ratio,
                )
            )
            stats["time"] = perf_counter() - start

        ecs.on("render", render)

    ecs.on("setup", setup)
