import math
from pyglm.glm import array, int32, normalize, vec2, vec3, vec4

from wgut.scene.mesh import Mesh


# TODO: fundamental problem with normals. Find a solution !
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

        position = vec4(x, base_y, z, 1.0)
        normal = normalize(vec3(x, radius / height, z))

        uv = vec2(i / segments, 0.0)
        tangent = vec3(-math.sin(angle), 0, math.cos(angle))
        bitangent = vec3(0, 1, 0)

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
        normal = normalize(vec3(nx, ny, nz))

        position = vec4(0.0, tip_y, 0.0, 1.0)
        uv = vec2(0.5, 1.0)
        tangent = vec3(-nz, 0, nx)
        bitangent = vec3(0, 1, 0)

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

        position = vec4(x, base_y, z, 1.0)
        normal = vec3(0, -1, 0)
        uv = vec2(0.5 + x / (2 * radius), 0.5 + z / (2 * radius))
        tangent = vec3(1, 0, 0)
        bitangent = vec3(0, 0, 1)

        idx = len(positions)

        positions.append(position)
        normals.append(normal)
        uvs.append(uv)
        tangents.append(tangent)
        bitangents.append(bitangent)

        base_disk_indices.append(idx)

    base_center_idx = len(positions)
    positions.append(vec4(0, base_y, 0, 1.0))
    normals.append(vec3(0, -1, 0))
    uvs.append(vec2(0.5, 0.5))
    tangents.append(vec3(1, 0, 0))
    bitangents.append(vec3(0, 0, 1))

    for i in range(segments):
        a = base_disk_indices[i]
        b = base_disk_indices[i + 1]
        c = base_center_idx
        indices += [a, b, c]

    return Mesh(
        array(positions),
        array(vec4(1.0)).repeat(len(positions)),
        array(uvs),
        array(normals),
        array(tangents),
        array(bitangents),
        array.from_numbers(int32, *indices),
    )
