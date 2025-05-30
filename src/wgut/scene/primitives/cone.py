import math
import numpy as np
from pyglm.glm import array, int32

from wgut.scene.mesh import Mesh, vertex


# TODO: Use GLM
def cone(radius=1.0, height=2.0, segments=32) -> Mesh:
    positions = []
    normals = []
    tangents = []
    bitangents = []
    uvs = []

    indices = []

    base_y = 0
    tip_y = height

    base_side_indices = []
    base_disk_indices = []
    tip_indices = []

    for i in range(segments + 1):
        angle = (i / segments) * 2 * math.pi
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius

        position = (x, base_y, z)

        nx = x
        ny = radius / height
        nz = z
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        normal = (nx / length, ny / length, nz / length)

        uv = (i / segments, 0.0)
        tangent = (-math.sin(angle), 0, math.cos(angle))
        bitangent = (0, 1, 0)

        idx = len(positions)

        positions.append(position)
        normals.append(normal)
        uvs.append(uv)
        tangents.append(tangent)
        bitangents.append(bitangent)

        base_side_indices.append(idx)

    for i in range(segments):
        angle = ((i + 0.5) / segments) * 2 * math.pi
        nx = math.cos(angle)
        nz = math.sin(angle)
        ny = radius / height
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        normal = (nx / length, ny / length, nz / length)

        position = (0.0, tip_y, 0.0)
        uv = (0.5, 1.0)
        tangent = (-nz, 0, nx)
        bitangent = (0, 1, 0)

        idx = len(positions)

        positions.append(position)
        normals.append(normal)
        uvs.append(uv)
        tangents.append(tangent)
        bitangents.append(bitangent)

        tip_indices.append(idx)

    for i in range(segments):
        a = tip_indices[i]
        b = base_side_indices[i]
        c = base_side_indices[i + 1]
        indices += [a, c, b]

    for i in range(segments + 1):
        angle = (i / segments) * 2 * math.pi
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius

        position = (x, base_y, z)
        normal = (0, -1, 0)
        uv = (0.5 + x / (2 * radius), 0.5 + z / (2 * radius))
        tangent = (1, 0, 0)
        bitangent = (0, 0, 1)

        idx = len(positions)

        positions.append(position)
        normals.append(normal)
        uvs.append(uv)
        tangents.append(tangent)
        bitangents.append(bitangent)

        base_disk_indices.append(idx)

    base_center_idx = len(positions)
    positions.append((0, base_y, 0))
    normals.append((0, -1, 0))
    uvs.append((0.5, 0.5))
    tangents.append((1, 0, 0))
    bitangents.append((0, 0, 1))

    for i in range(segments):
        a = base_disk_indices[i]
        b = base_disk_indices[i + 1]
        c = base_center_idx
        indices += [a, b, c]

    vertices = vertex(
        np.array(positions, dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        np.array(uvs, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(tangents, dtype=np.float32),
        np.array(bitangents, dtype=np.float32),
    )

    return Mesh(*vertices, array.from_numbers(int32, *indices))
