from dataclasses import dataclass
from wgpu import GPUTexture
from wgut.camera import Camera
from wgut.scene.ecs import ECS
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


def render_system(
    vertex_buffer_size: int,
    index_buffer_size: int,
    material_buffer_size: int,
    texture_array_size: tuple[int, int, int],
    texture_ids_buffer_size: int | None = None,
):
    renderer = Renderer(
        vertex_buffer_size,
        index_buffer_size,
        material_buffer_size,
        texture_array_size,
        texture_ids_buffer_size,
    )

    def render(ecs: ECS, screen: GPUTexture):
        cam_comp, _ = ecs.query_one([CameraComponent, ActiveCamera])
        assert isinstance(cam_comp, CameraComponent)
        camera = cam_comp.camera
        renderer.begin_frame()
        for mesh, material, transform in ecs.query(
            [Mesh, MaterialComponent, Transform]
        ):
            renderer.add_mesh(mesh, transform, material.material)
        renderer.end_frame(screen, camera)

    return render
