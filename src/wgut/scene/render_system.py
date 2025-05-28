from collections import defaultdict
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
    stats: dict


@dataclass
class Layer:
    name: str


def render_system(ecs: ECS, renderer: Renderer, layers: list[str | Layer]):
    if len(layers) == 0:
        raise ValueError("Must have at least one layer")

    for i in range(len(layers)):
        layer = layers[i]
        if isinstance(layer, Layer):
            layers[i] = layer.name

    def render(ecs: ECS, screen: GPUTexture):
        cam_comp: CameraComponent
        cam_comp, _ = ecs.query_one([CameraComponent, ActiveCamera])
        camera = cam_comp.camera
        layer_content = defaultdict(list)
        for mesh, material, transform, layer in ecs.query(
            [Mesh, MaterialComponent, Transform, Layer]
        ):
            layer_content[layer.name].append((mesh, transform, material))

        clear_color = True
        stats = {}
        for layer in layers:
            renderer.begin_frame()
            for mesh, transform, material in layer_content[layer]:
                renderer.add_mesh(mesh, transform, material.material)
            renderer.end_frame(
                screen, camera, clear_color=clear_color, clear_depth=True
            )
            stats[layer] = renderer.get_frame_stat()
            clear_color = False

        try:
            render_stat: RenderStat = ecs.query_one(RenderStat)
            render_stat.stats = stats
        except EntityNotFound:
            ecs.spawn([RenderStat(stats)])

    ecs.on("render", render)
