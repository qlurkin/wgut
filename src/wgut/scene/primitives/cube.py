from ..mesh import (
    Mesh,
    vertex,
)
import numpy as np


def cube(size=1.0) -> Mesh:
    hs = size / 2  # half-size

    faces = [
        # Front Face (+Z)
        {
            "normal": (0, 0, 1),
            "tangent": (1, 0, 0),
            "bitangent": (0, 1, 0),
            "verts": [
                (-hs, -hs, +hs, 0, 0),
                (+hs, -hs, +hs, 1, 0),
                (+hs, +hs, +hs, 1, 1),
                (-hs, +hs, +hs, 0, 1),
            ],
        },
        # Rear Face (-Z)
        {
            "normal": (0, 0, -1),
            "tangent": (-1, 0, 0),
            "bitangent": (0, 1, 0),
            "verts": [
                (+hs, -hs, -hs, 0, 0),
                (-hs, -hs, -hs, 1, 0),
                (-hs, +hs, -hs, 1, 1),
                (+hs, +hs, -hs, 0, 1),
            ],
        },
        # Left Face (-X)
        {
            "normal": (-1, 0, 0),
            "tangent": (0, 0, -1),
            "bitangent": (0, 1, 0),
            "verts": [
                (-hs, -hs, -hs, 0, 0),
                (-hs, -hs, +hs, 1, 0),
                (-hs, +hs, +hs, 1, 1),
                (-hs, +hs, -hs, 0, 1),
            ],
        },
        # Right Face (+X)
        {
            "normal": (1, 0, 0),
            "tangent": (0, 0, 1),
            "bitangent": (0, 1, 0),
            "verts": [
                (+hs, -hs, +hs, 0, 0),
                (+hs, -hs, -hs, 1, 0),
                (+hs, +hs, -hs, 1, 1),
                (+hs, +hs, +hs, 0, 1),
            ],
        },
        # Top Face (+Y)
        {
            "normal": (0, 1, 0),
            "tangent": (1, 0, 0),
            "bitangent": (0, 0, -1),
            "verts": [
                (-hs, +hs, +hs, 0, 0),
                (+hs, +hs, +hs, 1, 0),
                (+hs, +hs, -hs, 1, 1),
                (-hs, +hs, -hs, 0, 1),
            ],
        },
        # Bottom Face (-Y)
        {
            "normal": (0, -1, 0),
            "tangent": (1, 0, 0),
            "bitangent": (0, 0, 1),
            "verts": [
                (-hs, -hs, -hs, 0, 0),
                (+hs, -hs, -hs, 1, 0),
                (+hs, -hs, +hs, 1, 1),
                (-hs, -hs, +hs, 0, 1),
            ],
        },
    ]

    positions = []
    normals = []
    tangents = []
    bitangents = []
    uvs = []
    indices = []

    for _, face in enumerate(faces):
        normal = face["normal"]
        tangent = face["tangent"]
        bitangent = face["bitangent"]

        base_index = len(positions)
        for v in face["verts"]:
            pos = v[:3]
            uv = v[3:]

            positions.append(pos)
            normals.append(normal)
            uvs.append(uv)
            tangents.append(tangent)
            bitangents.append(bitangent)

        indices += [
            base_index + 0,
            base_index + 1,
            base_index + 2,
            base_index + 2,
            base_index + 3,
            base_index + 0,
        ]

    vertices = vertex(
        np.array(positions, dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        np.array(uvs, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(tangents, dtype=np.float32),
        np.array(bitangents, dtype=np.float32),
    )
    return Mesh(vertices, np.array(indices, dtype=np.int32))
