from dataclasses import dataclass
from wgpu import GPUTexture
from wgut.camera import Camera
from wgut.scene.ecs import ECS, EntityNotFound
from wgut.scene.material import Material
from wgut.scene.mesh import Mesh
from wgut.scene.renderer import Renderer
from wgut.scene.transform import Transform


class ActiveCamera:
    pass


@dataclass
class CameraComponent:
    camera: Camera


@dataclass
class MaterialComponent:
    material: Material


@dataclass
class RenderStat:
    stat: dict


@dataclass
class Layer:
    name: str
    enabled: bool = True


def render_system(ecs: ECS, renderer: Renderer, layers: list[str] = []):
    layers = ["default"] + layers + ["gizmo"]

    for layer in layers:
        ecs.spawn([Layer(layer)])

    def render(ecs: ECS, screen: GPUTexture):
        cam_comp: CameraComponent
        cam_comp, _ = ecs.query_one([CameraComponent, ActiveCamera])
        camera = cam_comp.camera
        renderer.begin_frame()
        for mesh, material, transform in ecs.query(
            [Mesh, MaterialComponent, Transform]
        ):
            renderer.add_mesh(mesh, transform, material.material)
        renderer.end_frame(screen, camera)
        try:
            render_stat: RenderStat = ecs.query_one(RenderStat)
            render_stat.stat = renderer.get_frame_stat()
        except EntityNotFound:
            stat = renderer.get_frame_stat()
            if stat is not None:
                ecs.spawn([RenderStat(stat)])

    ecs.on("render", render)
