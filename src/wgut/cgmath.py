import numpy as np
import numpy.typing as ntp
import math


def vec(*x: ntp.ArrayLike) -> ntp.NDArray:
    v = np.hstack(x).astype(np.float32)
    if v.ndim != 1:
        print(x)
        print(v)
        raise ValueError("Vec must have 1 dimention")
    return v


def vecn(n: int, *x: ntp.ArrayLike) -> ntp.NDArray:
    v = vec(*x)
    if v.size == 1:
        v = np.full(n, v[0], dtype=np.float32)
    if v.size != n:
        raise ValueError(f"Vec{n} must be of size {n}")
    return v


def vec2(*x: ntp.ArrayLike) -> ntp.NDArray:
    return vecn(2, *x)


def vec3(*x: ntp.ArrayLike) -> ntp.NDArray:
    return vecn(3, *x)


def vec4(*x: ntp.ArrayLike) -> ntp.NDArray:
    return vecn(4, *x)


def length(v: ntp.ArrayLike) -> float:
    return float(np.linalg.norm(v))


def normalize(v: ntp.ArrayLike) -> ntp.NDArray:
    v = vec(v)
    return v / length(v)


def from_homogenous(v: ntp.ArrayLike) -> ntp.NDArray:
    v = vec(v)
    return v[:-1] / v[-1]


def to_homogenous(v: ntp.ArrayLike) -> ntp.NDArray:
    v = vec(v)
    return np.hstack([v, [1.0]], dtype=np.float32)


def look_at(
    eye: ntp.ArrayLike, target: ntp.ArrayLike, up: ntp.ArrayLike
) -> ntp.NDArray:
    """
    Convert world coordinates to camera coordinates where the camera point in the direction of -z axis and up is the y axis.
    """
    eye = vec3(eye)
    target = vec3(target)
    up = vec3(up)
    dir = target - eye
    f = normalize(dir)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    # fmt: off
    return np.array([[ s[0],  s[1],  s[2], -eye @ s],
                     [ u[0],  u[1],  u[2], -eye @ u],
                     [-f[0], -f[1], -f[2],  eye @ f],
                     [    0,     0,     0,        1]], dtype=np.float32)
    # fmt: on


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> ntp.NDArray:
    """
    Convert camera coordinates to wgpu clip coordinates where x and y goes from -1.0 to 1.0 and z goes from 0.0 to 1.0.
    """
    angle = fovy_deg * math.pi / 180
    yspan = near * math.tan(angle)
    xspan = yspan * aspect

    c0r0 = 2 * near / xspan
    c1r1 = 2 * near / yspan
    c2r2 = -(far + near) / (far - near) / 2 - 0.5
    c3r2 = -far * near / (far - near)

    # fmt: off
    return np.array([[c0r0,    0,    0,    0],
                     [   0, c1r1,    0,    0],
                     [   0,    0, c2r2, c3r2],
                     [   0,    0,   -1,    0]], dtype=np.float32)
    # fmt: on


def rotation_matrix_from_axis_and_angle(
    axis: ntp.ArrayLike, angle: ntp.ArrayLike
) -> ntp.NDArray:
    axis = normalize(axis)

    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    R = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s, 0],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s, 0],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    return R
