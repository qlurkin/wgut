from collections import defaultdict
from dataclasses import dataclass
from pyglm.glm import array
from wgpu import GPUTexture
from wgut.camera import Camera
from wgut.scene.ecs import ECS, QueryOneWithNoResult
from wgut.scene.light import LightComponent
from wgut.scene.material import Material
from wgut.scene.mesh import Mesh
from wgut.scene.renderer import Renderer
from wgut.scene.transform import Transform


class ActiveCamera:
    pass


@dataclass
class CameraComponent:
    camera: Camera

    def __str__(self):
        return str(self.camera)


@dataclass
class MaterialComponent:
    material: Material

    def __str__(self):
        return str(self.material)

    def ecs_explorer_gui(self):
        if "ecs_explorer_gui" in dir(self.material):
            self.material.ecs_explorer_gui()  # type: ignore


@dataclass
class MeshComponent:
    mesh: Mesh

    def __str__(self):
        return str(self.mesh)


@dataclass
class RenderStat:
    stats: dict

    def __str__(self):
        return "RenderStat"


class Layer:
    def __init__(self, name: str = ""):
        self.name = name

    def __str__(self):
        return f"Layer(name={self.name})"


def render_system(ecs: ECS, renderer: Renderer, layers: list[Layer]):
    if len(layers) == 0:
        raise ValueError("Must have at least one layer")

    def render(ecs: ECS, screen: GPUTexture):
        cam_comp: CameraComponent
        cam_comp, _ = ecs.query_one([CameraComponent, ActiveCamera])
        camera = cam_comp.camera
        layer_content = defaultdict(list)
        for mesh, material, transform, layer in ecs.query(
            [MeshComponent, MaterialComponent, Transform, Layer]
        ):
            layer_content[layer].append((mesh, transform, material))

        lights_data = []
        for light, transform in ecs.query([LightComponent, Transform]):
            # 2 vec4 per light
            pos, color = light.light.get_data(transform.get_matrix())
            lights_data.append(pos)
            lights_data.append(color)

        lights = None
        if len(lights_data) != 0:
            lights = array(lights_data)

        clear_color = True
        stats = {}
        for layer in layers:
            renderer.begin_frame()
            for mesh, transform, material in layer_content[layer]:
                renderer.add_mesh(mesh.mesh, transform, material.material)
            renderer.end_frame(
                screen, camera, lights, clear_color=clear_color, clear_depth=True
            )
            clear_color = False
            stats[layer] = renderer.get_frame_stat()

        try:
            render_stat: RenderStat = ecs.query_one(RenderStat)
            render_stat.stats = stats
        except QueryOneWithNoResult:
            ecs.spawn([RenderStat(stats)])

    ecs.on("render", render)
