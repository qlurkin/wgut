from pyglm.glm import (
    array,
    dot,
    mat3,
    normalize,
    scale,
    vec3,
    vec4,
    length,
    min,
)
from wgut.orbit_camera import OrbitCamera
from wgut.scene.instance_mesh import InstanceMesh
from wgut.scene.performance_monitor import performance_monitor
from wgut.scene.render_gui_system import render_gui_system
from wgut.scene.render_system import (
    ActiveCamera,
    CameraComponent,
    Layer,
    MaterialComponent,
    MeshComponent,
    render_system,
)
from wgut.scene.renderer import Renderer
from wgut.scene.window_system import window_system
from wgut.scene.ecs import ECS
from wgut.scene.basic_color_material import BasicColorMaterial
from wgut.scene.primitives.icosphere import icosphere
from wgut.scene.transform import Transform
import numpy as np

default_layer = Layer("default")


def tensor_product(v1: vec3, v2: vec3):
    return mat3(v1 * v2.x, v1 * v2.y, v1 * v2.z)  # type: ignore


class Strand:
    def __init__(
        self,
        positions: array[vec3],
        mass: float,
        stiffness: float,
    ):
        num_verts = len(positions)
        self.__vert_mass = mass / num_verts
        self.__stiffness = stiffness

        self.__positions = positions
        self.__velocities = array.zeros(num_verts, vec3)
        self.__edges = []
        self.__lengthes = []
        self.__vert_adjacent_edges = [[] for _ in range(num_verts)]
        self.__substep = 10
        self.__num_iterations = 10
        self.__gravity = vec3(0.0, -9.81, 0.0)
        self.__inertia = array(self.__positions)

        for i in range(num_verts - 1):
            edge_id = len(self.__edges)
            le = length(self.__positions[i] - self.__positions[i + 1])
            self.__lengthes.append(le)
            self.__edges.append([i, i + 1])

            self.__vert_adjacent_edges[i].append(edge_id)
            self.__vert_adjacent_edges[i + 1].append(edge_id)

        self.__prev_positions = array(self.__positions)

    def get_positions(self) -> array[vec4]:
        return self.__positions.map(lambda pos: vec4(pos.x, pos.y, pos.z, 1.0))

    def forward_step(self, dt: float):
        self.__inertia = self.__positions + dt * self.__velocities
        self.__prev_positions = array(self.__positions)
        self.__positions = array(self.__inertia)

    def solve(self, dt: float):
        dt_sqr_reciprocal = 1 / dt / dt

        for i in list(range(1, len(self.__positions))):
            f = (
                self.__vert_mass
                * (self.__inertia[i] - self.__positions[i])
                * dt_sqr_reciprocal
            )
            f += self.__vert_mass * self.__gravity
            h = mat3(self.__vert_mass * dt_sqr_reciprocal)

            for edge_id in self.__vert_adjacent_edges[i]:
                if self.__edges[edge_id][0] == i:
                    o = self.__edges[edge_id][1]
                else:
                    o = self.__edges[edge_id][0]

                diff = self.__positions[i] - self.__positions[o]
                l = length(diff)
                l0 = self.__lengthes[edge_id]

                # evaluate hessian
                h_1_1 = self.__stiffness * (
                    mat3() - (l0 / l) * (mat3() - tensor_product(diff, diff) / (l * l))
                )
                h += h_1_1

                f += (self.__stiffness * (l0 - l) / l) * diff

            dx = np.linalg.lstsq(np.array(h), np.array(f), rcond=None)[0]
            self.__positions[i] += vec3(dx)

    def update_velocity(self, dt: float):
        self.__velocities = (self.__positions - self.__prev_positions) / dt
        self.__velocities[0] = vec3(0.0)

    def update(self, ecs: ECS, delta_time: float):
        if delta_time == 0.0:
            return
        dt = delta_time / self.__substep
        for _ in range(self.__substep):
            self.forward_step(dt)
            for _ in range(self.__num_iterations):
                self.solve(dt)
            self.update_velocity(dt)


def setup(ecs: ECS):
    mesh = icosphere(2)

    positions = []
    for i in range(5):
        positions.append(vec3(i * 1, 2.5, 0.0))

    strand = Strand(array(positions), 0.001, 100)

    particle_mesh = InstanceMesh(mesh, strand.get_positions)

    ecs.on("update", strand.update)

    material = BasicColorMaterial((1.0, 0.0, 0.0))

    ecs.spawn(
        [
            MaterialComponent(material),
            MeshComponent(particle_mesh),
            Transform(scale(vec3(0.2))),
            default_layer,
        ]
    )

    camera = OrbitCamera((0, 0, -5), (0, 0, 0), 45, 0.1, 1000)

    ecs.spawn([CameraComponent(camera), ActiveCamera()], label="Camera")


def process_event(ecs: ECS, event):
    cam: CameraComponent = ecs.query_one(CameraComponent)
    if isinstance(cam.camera, OrbitCamera):
        cam.camera.process_event(event)


renderer = Renderer(30000, 150000, 512, 2, (1024, 1024, 32), 256)


(
    ECS()
    .on("setup", setup)
    .on("window_event", process_event)
    .do(performance_monitor)
    .do(
        render_system,
        renderer,
        [default_layer],
    )
    .do(render_gui_system)
    .do(window_system, "Hello Particles")
)
