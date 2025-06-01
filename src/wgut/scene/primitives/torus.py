import math

from pyglm.glm import array, cross, int32, vec2, vec3, vec4
from ..mesh import Mesh


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
            position = vec4(x, y, z, 1.0)

            nx = cos_phi * cos_theta
            ny = cos_phi * sin_theta
            nz = sin_phi
            normal = vec3(nx, ny, nz)

            tx = -sin_theta
            ty = cos_theta
            tz = 0
            tangent = vec3(tx, ty, tz)

            bitangent = cross(normal, tangent)

            positions.append(position)
            normals.append(normal)
            uvs.append(vec2(u, v))
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

    return Mesh(
        array(positions),
        array(vec4(1.0)).repeat(len(positions)),
        array(uvs),
        array(normals),
        array(tangents),
        array(bitangents),
        array.from_numbers(int32, *indices),
    )
