from pyglm.glm import array, int32, vec2, vec3, vec4
from ..mesh import Mesh


# TODO: Use GLM
def cube(size=1.0) -> Mesh:
    hs = size / 2  # half-size

    faces = [
        # Front Face (+Z)
        {
            "normal": vec3(0, 0, 1),
            "tangent": vec3(1, 0, 0),
            "bitangent": vec3(0, 1, 0),
            "positions": array(
                vec4(-hs, -hs, +hs, 1.0),
                vec4(+hs, -hs, +hs, 1.0),
                vec4(+hs, +hs, +hs, 1.0),
                vec4(-hs, +hs, +hs, 1.0),
            ),
            "uvs": array(vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)),
        },
        # Rear Face (-Z)
        {
            "normal": vec3(0, 0, -1),
            "tangent": vec3(-1, 0, 0),
            "bitangent": vec3(0, 1, 0),
            "positions": array(
                vec4(+hs, -hs, -hs, 1.0),
                vec4(-hs, -hs, -hs, 1.0),
                vec4(-hs, +hs, -hs, 1.0),
                vec4(+hs, +hs, -hs, 1.0),
            ),
            "uvs": array(vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)),
        },
        # Left Face (-X)
        {
            "normal": vec3(-1, 0, 0),
            "tangent": vec3(0, 0, -1),
            "bitangent": vec3(0, 1, 0),
            "positions": array(
                vec4(-hs, -hs, -hs, 1.0),
                vec4(-hs, -hs, +hs, 1.0),
                vec4(-hs, +hs, +hs, 1.0),
                vec4(-hs, +hs, -hs, 1.0),
            ),
            "uvs": array(vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)),
        },
        # Right Face (+X)
        {
            "normal": vec3(1, 0, 0),
            "tangent": vec3(0, 0, 1),
            "bitangent": vec3(0, 1, 0),
            "positions": array(
                vec4(+hs, -hs, +hs, 1.0),
                vec4(+hs, -hs, -hs, 1.0),
                vec4(+hs, +hs, -hs, 1.0),
                vec4(+hs, +hs, +hs, 1.0),
            ),
            "uvs": array(vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)),
        },
        # Top Face (+Y)
        {
            "normal": vec3(0, 1, 0),
            "tangent": vec3(1, 0, 0),
            "bitangent": vec3(0, 0, -1),
            "positions": array(
                vec4(-hs, +hs, +hs, 1.0),
                vec4(+hs, +hs, +hs, 1.0),
                vec4(+hs, +hs, -hs, 1.0),
                vec4(-hs, +hs, -hs, 1.0),
            ),
            "uvs": array(vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)),
        },
        # Bottom Face (-Y)
        {
            "normal": vec3(0, -1, 0),
            "tangent": vec3(1, 0, 0),
            "bitangent": vec3(0, 0, 1),
            "positions": array(
                vec4(-hs, -hs, -hs, 1.0),
                vec4(+hs, -hs, -hs, 1.0),
                vec4(+hs, -hs, +hs, 1.0),
                vec4(-hs, -hs, +hs, 1.0),
            ),
            "uvs": array(vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)),
        },
    ]

    positions = []
    normals = []
    tangents = []
    bitangents = []
    uvs = []
    indices = []

    for face in faces:
        normal = face["normal"]
        tangent = face["tangent"]
        bitangent = face["bitangent"]

        base_index = len(positions)
        for i in range(4):
            pos = face["positions"][i]
            uv = face["uvs"][i]

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

    return Mesh(
        array(positions),
        array(vec4(1.0)).repeat(len(positions)),
        array(uvs),
        array(normals),
        array(tangents),
        array(bitangents),
        array.from_numbers(int32, *indices),
    )
