from dataclasses import dataclass
from typing import Callable
from pygfx import (
    Camera,
    WgpuRenderer,
    WorldObject,
    Scene,
)
from wgut.scene.ecs import ECS
from wgut.window import Window


class ActiveCamera:
    def __str__(self):
        return "ActiveCamera"


@dataclass
class SceneObject:
    obj: WorldObject

    def __str__(self):
        return str(self.obj)


def render_system(ecs: ECS):
    renderer = None
    scene = Scene()
    wait_for_renderer = []

    def handle(ecs: ECS, fn: Callable[[WgpuRenderer], None]):
        if renderer is None:
            wait_for_renderer.append(fn)
        else:
            fn(renderer)

    def setup(ecs: ECS, window: Window):
        nonlocal renderer
        canvas = window.get_canvas()
        renderer = WgpuRenderer(target=canvas)

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
        )

        for fn in wait_for_renderer:
            fn(renderer)

    def render(ecs: ECS):
        if renderer is None:
            return

        cam_so: SceneObject
        cam_so, _ = ecs.query_one([SceneObject, ActiveCamera])
        camera = cam_so.obj

        assert isinstance(camera, Camera), "The ActiveCamera is not a Camera"

        scene.clear()
        for (so,) in ecs.query([SceneObject]):
            scene.add(so.obj)

        renderer.clear(all=True)
        renderer.render(scene, camera)

    ecs.on("render", render)
    ecs.on("setup", setup)
    ecs.on("call_with_renderer", handle)
