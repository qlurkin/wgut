import math
import numpy as np

from wgut.scene.mesh import Mesh, vertex


def cylinder(radius=1.0, height=2.0, segments=32):
    positions = []
    normals = []
    uvs = []
    tangents = []
    bitangents = []

    indices = []

    half_height = height / 2
    top_y = +half_height
    bottom_y = -half_height

    side_top_indices = []
    side_bottom_indices = []
    cap_top_indices = []
    cap_bottom_indices = []

    # ----- Sides -----
    for i in range(segments + 1):
        angle = (i / segments) * 2 * math.pi
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius

        tangent = (-math.sin(angle), 0, math.cos(angle))
        bitangent = (0, 1, 0)
        normal = (x, 0, z)
        length = math.sqrt(x * x + z * z)
        normal = (x / length, 0, z / length)

        u = i / segments

        idx = len(positions)

        positions.append((x, bottom_y, z))
        normals.append(normal)
        uvs.append((u, 0.0))
        tangents.append(tangent)
        bitangents.append(bitangent)

        side_bottom_indices.append(idx)

        idx = len(positions)

        positions.append((x, top_y, z))
        normals.append(normal)
        uvs.append((u, 1.0))
        tangents.append(tangent)
        bitangents.append(bitangent)

        side_top_indices.append(idx)

    for i in range(segments):
        b0 = side_bottom_indices[i]
        b1 = side_bottom_indices[i + 1]
        t0 = side_top_indices[i]
        t1 = side_top_indices[i + 1]

        indices += [b0, t0, t1]
        indices += [b0, t1, b1]

    # ----- Top -----
    for i in range(segments + 1):
        angle = (i / segments) * 2 * math.pi
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        u = 0.5 + x / (2 * radius)
        v = 0.5 + z / (2 * radius)

        idx = len(positions)

        positions.append((x, top_y, z))
        normals.append((0, 1, 0))
        uvs.append((u, v))
        tangents.append((1, 0, 0))
        bitangents.append((0, 0, 1))

        cap_top_indices.append(idx)

    top_center_idx = len(positions)

    positions.append((0, top_y, 0))
    normals.append((0, 1, 0))
    uvs.append((0.5, 0.5))
    tangents.append((1, 0, 0))
    bitangents.append((0, 0, 1))

    for i in range(segments):
        a = cap_top_indices[i]
        b = cap_top_indices[i + 1]
        c = top_center_idx
        indices += [a, c, b]

    # ----- Bottom -----
    for i in range(segments + 1):
        angle = (i / segments) * 2 * math.pi
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        u = 0.5 + x / (2 * radius)
        v = 0.5 + z / (2 * radius)

        idx = len(positions)

        positions.append((x, bottom_y, z))
        normals.append((0, -1, 0))
        uvs.append((u, v))
        tangents.append((1, 0, 0))
        bitangents.append((0, 0, 1))

        cap_bottom_indices.append(idx)

    bottom_center_idx = len(positions)

    positions.append((0, bottom_y, 0))
    normals.append((0, -1, 0))
    uvs.append((0.5, 0.5))
    tangents.append((1, 0, 0))
    bitangents.append((0, 0, 1))

    for i in range(segments):
        a = cap_bottom_indices[i]
        b = cap_bottom_indices[i + 1]
        c = bottom_center_idx
        indices += [a, b, c]

    vertices = vertex(
        np.array(positions, dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        np.array(uvs, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(tangents, dtype=np.float32),
        np.array(bitangents, dtype=np.float32),
    )
    return Mesh(vertices, np.array(indices, dtype=np.int32))
