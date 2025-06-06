from typing import Callable
from pyglm.glm import array, int32, mat4, vec2, vec3, vec4
from wgpu import BufferUsage, GPUBuffer

from wgut.builders.bufferbuilder import BufferBuilder
from wgut.core import read_buffer
from wgut.scene.ecs import ECS
from wgut.scene.mesh import Mesh


class Particles:
    def __init__(
        self,
        mesh: Mesh,
        translations: array[vec4],
        callback: Callable[[ECS, GPUBuffer, float], None],
    ):
        self.__mesh = mesh
        self.__translations = (
            BufferBuilder()
            .from_data(translations)
            .with_usage(BufferUsage.STORAGE | BufferUsage.COPY_SRC)
            .build()
        )
        self.__callback = callback
        self.__particle_count = len(translations)
        self.__vertex_count = 0
        self.__index_count = 0
        self.__positions: array[vec4]
        self.__colors: array[vec4]
        self.__uvs: array[vec2]
        self.__normals: array[vec3]
        self.__tangents: array[vec3]
        self.__bitangents: array[vec3]
        self.__indices: array[int32]

    def update(self, ecs: ECS, delta_time: float):
        self.__callback(ecs, self.__translations, delta_time)

    def get_transformed_vertices(
        self, transformation_matrix: mat4
    ) -> tuple[
        array[vec4], array[vec4], array[vec2], array[vec3], array[vec3], array[vec3]
    ]:
        mem = read_buffer(self.__translations)
        translations = array.as_reference(mem).reinterpret_cast(vec4)  # type: ignore

        (
            mesh_positions,
            mesh_colors,
            mesh_uvs,
            mesh_normals,
            mesh_tangents,
            mesh_bitangents,
        ) = self.__mesh.get_transformed_vertices(transformation_matrix)
        mesh_indices = self.__mesh.get_indices()
        mesh_vertex_count = len(mesh_positions)
        mesh_index_count = len(mesh_indices)

        vertex_count = self.__particle_count * mesh_vertex_count
        if vertex_count != self.__vertex_count:
            self.__positions = array(vec4(0.0)).repeat(vertex_count)
            self.__colors = array(vec4(0.0)).repeat(vertex_count)
            self.__uvs = array(vec2(0.0)).repeat(vertex_count)
            self.__normals = array(vec3(0.0)).repeat(vertex_count)
            self.__tangents = array(vec3(0.0)).repeat(vertex_count)
            self.__bitangents = array(vec3(0.0)).repeat(vertex_count)

        index_count = self.__particle_count * mesh_index_count
        if index_count != self.__index_count:
            self.__indices = array(int32(0)).repeat(index_count)

        for i, translation in enumerate(translations):
            self.__positions[
                i * mesh_vertex_count : i * mesh_vertex_count + mesh_vertex_count
            ] = mesh_positions + translation
            self.__colors[
                i * mesh_vertex_count : i * mesh_vertex_count + mesh_vertex_count
            ] = mesh_colors
            self.__uvs[
                i * mesh_vertex_count : i * mesh_vertex_count + mesh_vertex_count
            ] = mesh_uvs
            self.__normals[
                i * mesh_vertex_count : i * mesh_vertex_count + mesh_vertex_count
            ] = mesh_normals
            self.__tangents[
                i * mesh_vertex_count : i * mesh_vertex_count + mesh_vertex_count
            ] = mesh_tangents
            self.__bitangents[
                i * mesh_vertex_count : i * mesh_vertex_count + mesh_vertex_count
            ] = mesh_bitangents
            self.__indices[
                i * mesh_index_count : i * mesh_index_count + mesh_index_count
            ] = mesh_indices + i * mesh_vertex_count

        return (
            self.__positions,
            self.__colors,
            self.__uvs,
            self.__normals,
            self.__tangents,
            self.__bitangents,
        )

    def get_indices(self) -> array[int32]:
        return self.__indices


if __name__ == "__main__":
    from wgut.scene.primitives.icosphere import icosphere
    from pyglm.glm import identity

    mesh = icosphere(0)

    def callback(ecs: ECS, buffer: GPUBuffer, delta_time: float):
        pass

    particles = Particles(
        mesh,
        array(
            vec4(1, 0, 1, 0), vec4(-1, 0, 1, 0), vec4(1, 0, -1, 0), vec4(-1, 0, -1, 0)
        ),
        callback,
    )

    ecs = ECS()

    particles.update(ecs, 0.016)

    positions, colors, uvs, normals, tangents, bitangents = (
        particles.get_transformed_vertices(identity(mat4))
    )
    indices = particles.get_indices()

    print(positions)
