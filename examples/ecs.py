from imgui_bundle import imgui
from wgut.orbit_camera import OrbitCamera
from wgut.scene.render_gui_system import render_gui_system
from wgut.scene.render_system import (
    ActiveCamera,
    CameraComponent,
    MaterialComponent,
    RenderStat,
    render_system,
)
from wgut.scene.window_system import window_system
from wgut.scene.ecs import ECS, EntityNotFound
from wgut.scene.pbr_material import PbrMaterial
from wgut.scene.primitives.icosphere import icosphere
from wgut.scene.transform import Transform


def setup(ecs: ECS):
    mesh = icosphere(4)
    material = PbrMaterial(
        "./textures/Wood_025_basecolor.jpg",
        "./textures/Wood_025_normal.jpg",
        "./textures/Wood_025_roughness.jpg",
        0.0,
        (0.0, 0.0, 0.0),
        None,
    )
    transform = Transform()
    camera = OrbitCamera((6, 4, 5), (0, 0, 0), 45, 0.1, 100)
    ecs.spawn([mesh, MaterialComponent(material), transform])
    ecs.spawn([CameraComponent(camera), ActiveCamera()])


def process_event(ecs: ECS, event):
    (cam,) = ecs.query_one([CameraComponent])
    cam.camera.process_event(event)


def gui(ecs: ECS) -> imgui.ImDrawData:
    try:
        (r_stat,) = ecs.query_one([RenderStat])
        stat = r_stat.stat
        imgui.new_frame()
        imgui.begin("Scene", None)
        imgui.text(f"Render Time: {stat['time']:.5f}s")
        imgui.text(f"Draw count: {stat['draw']}")
        imgui.text(f"Mesh count: {stat['mesh']}")
        imgui.text(f"Triangle count: {stat['triangle']}")
        imgui.text(f"Vertex count: {stat['vertex']}")
        imgui.end()
        imgui.end_frame()
        imgui.render()
    except EntityNotFound:
        pass
    return imgui.get_draw_data()


(
    ECS()
    .on("setup", setup)
    .on("window_event", process_event)
    .on("setup", render_gui_system(gui))
    .on("render", render_system(10000, 50000, 128, (1024, 1024, 7), 48))
    .do(window_system)
)
