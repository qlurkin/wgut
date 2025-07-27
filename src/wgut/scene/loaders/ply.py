from pyglm.glm import vec2, vec3, vec4, array, int32
from typing import List
import re

from wgut.scene.static_mesh import StaticMesh


def load_ply(ply_path: str) -> List[List[object]]:
    with open(ply_path, "r") as f:
        lines = f.readlines()

    # === 1. Parse header ===
    header_ended = False
    vertex_count = 0
    face_count = 0
    properties = []
    reading_vertex = False
    reading_face = False

    i = 0
    while not header_ended and i < len(lines):
        line = lines[i].strip()
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
            reading_vertex = True
        elif line.startswith("element face"):
            face_count = int(line.split()[-1])
            reading_vertex = False
            reading_face = True
        elif line.startswith("property") and reading_vertex:
            properties.append(line.split()[2])  # ex: x, y, z, nx, ny, ...
        elif line.startswith("end_header"):
            header_ended = True
        i += 1

    # === 2. Read vertex data ===
    positions = []
    normals = []
    colors = []
    uvs = []
    tangents = []
    bitangents = []

    for j in range(vertex_count):
        tokens = lines[i + j].strip().split()
        prop_dict = dict(zip(properties, tokens))

        x, y, z = float(prop_dict["x"]), float(prop_dict["y"]), float(prop_dict["z"])
        positions.append(vec4(x, y, z, 1.0))

        # Normals
        nx = float(prop_dict.get("nx", 0.0))
        ny = float(prop_dict.get("ny", 0.0))
        nz = float(prop_dict.get("nz", 0.0))
        normals.append(vec3(nx, ny, nz))

        # UVs
        u = float(prop_dict.get("u", prop_dict.get("texture_u", 0.0)))
        v = float(prop_dict.get("v", prop_dict.get("texture_v", 0.0)))
        uvs.append(vec2(u, v))

        # Colors
        r = float(prop_dict.get("red", 255)) / 255.0
        g = float(prop_dict.get("green", 255)) / 255.0
        b = float(prop_dict.get("blue", 255)) / 255.0
        a = float(prop_dict.get("alpha", 255)) / 255.0
        colors.append(vec4(r, g, b, a))

        # Placeholder tangent space
        tangents.append(vec3(1, 0, 0))
        bitangents.append(vec3(0, 1, 0))

    # === 3. Read face indices ===
    indices = []
    for j in range(face_count):
        line = lines[i + vertex_count + j].strip()
        match = re.match(r"^(\d+)\s+(.*)", line)
        if not match:
            continue
        count = int(match.group(1))
        vertex_indices = list(map(int, match.group(2).split()))
        if count >= 3:
            for k in range(1, count - 1):
                indices += [vertex_indices[0], vertex_indices[k], vertex_indices[k + 1]]

    mesh = StaticMesh(
        array(positions),
        array(colors),
        array(uvs),
        array(normals),
        array(tangents),
        array(bitangents),
        array.from_numbers(int32, *indices),
    )

    return [[mesh]]
