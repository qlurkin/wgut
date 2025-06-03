from typing import Any
from pyglm.glm import int32, vec2, vec3, vec4, array
from wgut.scene.mesh import Mesh, compute_bitangent_vectors, compute_tangent_vectors
import os
from collections import namedtuple, defaultdict
from wgut.scene.pbr_material import PbrMaterial
from wgut.scene.render_system import MaterialComponent


VertexKey = namedtuple("VertexKey", ["v", "vt", "vn", "vc"])


def parse_mtl_file(mtl_path: str) -> dict:
    materials = {}
    current = None

    def parse_color(tokens):
        if len(tokens) == 4:
            return tuple(map(float, tokens[1:5]))  # RGBA
        elif len(tokens) == 3:
            return tuple(map(float, tokens[1:4])) + (1.0,)
        return (1.0, 1.0, 1.0, 1.0)

    with open(mtl_path, "r") as f:
        for line in f:
            if line.startswith("newmtl"):
                current = line.strip().split()[1]
                materials[current] = {}
            elif current:
                tokens = line.strip().split()
                if not tokens:
                    continue
                key = tokens[0]
                if key == "Kd":
                    materials[current]["albedo"] = parse_color(tokens)
                elif key == "map_Kd":
                    materials[current]["albedo"] = tokens[1]
                elif key in ["map_Bump", "bump"]:
                    materials[current]["normal"] = tokens[1]
                elif key == "map_Roughness":
                    materials[current]["roughness"] = tokens[1]
                elif key == "map_Metallic":
                    materials[current]["metallicity"] = tokens[1]
                elif key == "map_Occlusion":
                    materials[current]["occlusion"] = tokens[1]
                elif key == "Ke":
                    materials[current]["emissivity"] = tuple(map(float, tokens[1:4]))

    return materials


# TODO: support for no UVs (impossible to compute tangents) or no normals (compute them)


def load_obj(obj_path: str) -> list[list[Any]]:
    positions_raw = []
    uvs_raw = []
    normals_raw = []
    colors_raw = []

    material_groups = defaultdict(list)
    current_material = "default"

    materials = {}

    base_dir = os.path.dirname(obj_path)

    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("mtllib"):
                mtl_file = line.strip().split()[1]
                mtl_path = os.path.join(base_dir, mtl_file)
                materials = parse_mtl_file(mtl_path)
            elif line.startswith("usemtl"):
                current_material = line.strip().split()[1]
            elif line.startswith("v "):
                tokens = list(map(float, line.strip().split()[1:]))
                positions_raw.append(vec3(*tokens[:3]))
                if len(tokens) == 6:
                    colors_raw.append(vec4(tokens[3], tokens[4], tokens[5], 1.0))
                else:
                    colors_raw.append(vec4(1.0))  # default white
            elif line.startswith("vt "):
                _, u, v = line.strip().split()
                uvs_raw.append(vec2(float(u), float(v)))
            elif line.startswith("vn "):
                _, nx, ny, nz = line.strip().split()
                normals_raw.append(vec3(float(nx), float(ny), float(nz)))
            elif line.startswith("f "):
                face = line.strip().split()[1:]
                group = material_groups[current_material]
                group.append(face)

    results = []

    for material_name, faces in material_groups.items():
        vertex_map = {}
        vertices = []
        indices = []
        idx_counter = 0

        for face in faces:
            face_indices = []
            for v in face:
                parts = v.split("/")
                vi = int(parts[0]) - 1
                vti = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else None
                vni = int(parts[2]) - 1 if len(parts) > 2 and parts[2] else None

                key = VertexKey(vi, vti, vni, vi)  # `vi` used for color too

                if key not in vertex_map:
                    vertex_map[key] = idx_counter
                    vertices.append(key)
                    idx_counter += 1

                face_indices.append(vertex_map[key])

            for i in range(1, len(face_indices) - 1):
                indices.extend([face_indices[0], face_indices[i], face_indices[i + 1]])

        pos_array = []
        uv_array = []
        norm_array = []
        color_array = []
        tangent_array = []
        bitangent_array = []

        for key in vertices:
            pos_array.append(vec4(positions_raw[key.v], 1.0))  # type: ignore
            color_array.append(colors_raw[key.vc])
            uv_array.append(uvs_raw[key.vt] if key.vt is not None else vec2(0.0))
            norm_array.append(normals_raw[key.vn] if key.vn is not None else vec3(0.0))
            tangent_array.append(vec3(1, 0, 0))  # placeholder
            bitangent_array.append(vec3(0, 1, 0))  # placeholder

        positions = array(pos_array)
        colors = array(color_array)
        uvs = array(uv_array)
        normals = array(norm_array)
        indices = array.from_numbers(int32, *indices)
        tangents = compute_tangent_vectors(positions, uvs, normals, indices)
        bitangents = compute_bitangent_vectors(normals, tangents)

        mesh = Mesh(
            positions,
            colors,
            uvs,
            normals,
            tangents,
            bitangents,
            indices,
        )

        components: list[Any] = [mesh]

        if material_name in materials:
            mat_data = materials[material_name]
            material = PbrMaterial(
                albedo=mat_data.get("albedo", (1.0, 1.0, 1.0, 1.0)),
                normal=mat_data.get("normal", None),
                roughness=mat_data.get("roughness", 1.0),
                metalicity=mat_data.get("metallicity", 0.0),
                emissivity=mat_data.get("emissivity", (0.0, 0.0, 0.0)),
                occlusion=mat_data.get("occlusion", None),
            )
            components.append(MaterialComponent(material))

        results.append(components)

    return results
