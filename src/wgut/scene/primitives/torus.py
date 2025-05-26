import math
from ..mesh import (
    Mesh,
    vertex,
)
import numpy as np


def torus(
    radius_major=1.0, radius_minor=0.3, segments_major=32, segments_minor=16
) -> Mesh:
    positions = []
    normals = []
    tangents = []
    bitangents = []
    uvs = []
    indices = []

    for i in range(segments_major + 1):
        u = i / segments_major
        theta = u * 2 * math.pi
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        for j in range(segments_minor + 1):
            v = j / segments_minor
            phi = v * 2 * math.pi
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)

            x = (radius_major + radius_minor * cos_phi) * cos_theta
            y = (radius_major + radius_minor * cos_phi) * sin_theta
            z = radius_minor * sin_phi
            position = (x, y, z)

            nx = cos_phi * cos_theta
            ny = cos_phi * sin_theta
            nz = sin_phi
            normal = (nx, ny, nz)

            tx = -sin_theta
            ty = cos_theta
            tz = 0
            tangent = (tx, ty, tz)

            btx = normal[1] * tangent[2] - normal[2] * tangent[1]
            bty = normal[2] * tangent[0] - normal[0] * tangent[2]
            btz = normal[0] * tangent[1] - normal[1] * tangent[0]
            bitangent = (btx, bty, btz)

            positions.append(position)
            normals.append(normal)
            uvs.append((u, v))
            tangents.append(tangent)
            bitangents.append(bitangent)

    verts_per_row = segments_minor + 1
    for i in range(segments_major):
        for j in range(segments_minor):
            a = i * verts_per_row + j
            b = (i + 1) * verts_per_row + j
            c = (i + 1) * verts_per_row + (j + 1)
            d = i * verts_per_row + (j + 1)

            indices += [a, b, d]
            indices += [b, c, d]

    vertices = vertex(
        np.array(positions, dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        np.array(uvs, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(tangents, dtype=np.float32),
        np.array(bitangents, dtype=np.float32),
    )
    return Mesh(vertices, np.array(indices, dtype=np.int32))
