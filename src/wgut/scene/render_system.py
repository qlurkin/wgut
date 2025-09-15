from dataclasses import dataclass
from wgpu import GPUTexture
from wgut.camera import Camera
from wgut.scene.ecs import ECS, QueryOneWithNoResult
from wgut.scene.light import LightComponent
from wgut.scene.mesh import Mesh
from wgut.scene.renderer import Renderer, Material
from wgut.scene.transform import Transform
import numpy as np


class ActiveCamera:
    def __str__(self):
        return "ActiveCamera"


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
class RenderStats:
    stats: dict

    def __str__(self):
        return "RenderStat"


def render_system(ecs: ECS, renderer: Renderer):
    def render(ecs: ECS, screen: GPUTexture):
        cam_comp: CameraComponent
        cam_comp, _ = ecs.query_one([CameraComponent, ActiveCamera])
        camera = cam_comp.camera
        view_matrix, proj_matrix = camera.get_matrices(screen.width / screen.height)

        camera_position = np.hstack([camera.get_position(), [1.0]]).astype(np.float32)
        camera_matrix = np.array(proj_matrix @ view_matrix, dtype=np.float32)

        lights_data = []
        for light, transform in ecs.query([LightComponent, Transform]):
            # 2 vec4 per light
            pos, color = light.light.get_data(transform.get_matrix())
            lights_data.append(pos)
            lights_data.append(color)

        lights = None
        if len(lights_data) != 0:
            lights = np.array(lights_data)

        renderer.clear_color(screen, (0.9, 0.9, 0.9, 1.0))
        renderer.clear_depth()
        renderer.begin_frame(
            screen,
            camera_matrix,
            camera_position,
            lights,
        )
        for mesh, transform, material in ecs.query(
            [MeshComponent, Transform, MaterialComponent]
        ):
            renderer.add_mesh(mesh.mesh, transform, material.material)
        renderer.end_frame()
        stats = renderer.get_frame_stat()

        try:
            render_stats: RenderStats = ecs.query_one(RenderStats)
            render_stats.stats = stats
        except QueryOneWithNoResult:
            ecs.spawn([RenderStats(stats)])

    ecs.on("render", render)
