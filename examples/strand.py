from pyglm.glm import (
    array,
    dot,
    float32,
    mat3,
    normalize,
    scale,
    vec3,
    vec4,
    i32vec2,
    length,
    min,
)
from wgpu import BufferUsage, GPUBuffer
from wgut.auto_compute_pipeline import AutoComputePipeline
from wgut.builders.bufferbuilder import BufferBuilder
from wgut.core import load_file, write_buffer
from wgut.orbit_camera import OrbitCamera
from wgut.scene.instance_mesh import InstanceMesh
from wgut.scene.particles import Particles
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
import random
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
        self.__substep = 1
        self.__num_iterations = 100
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
        self.__prev_velocities = array(self.__velocities)

        print(self.__vert_adjacent_edges)

    def get_positions(self) -> array[vec4]:
        return self.__positions.map(lambda pos: vec4(pos.x, pos.y, pos.z, 1.0))

    # TODO: Seems to be optimisable
    def forward_step(self, dt: float):
        self.__velocities = self.__velocities.map(lambda v: v + dt * self.__gravity)
        self.__velocities[0] = vec3(0.0)  # fix first vertex to anchor the strand

        self.__inertia = self.__positions + dt * self.__velocities
        self.__prev_positions = array(self.__positions)

        acceleration_approx = (self.__velocities - self.__prev_velocities) / dt

        grav_norm = length(self.__gravity)
        grav_dir = normalize(self.__gravity)

        # acceleration_component = acceleration_approx.map(lambda aa: dot(aa, grav_dir))
        #
        # acceleration_component = acceleration_component.map(
        #     lambda ac: min(ac, grav_norm)
        # )
        #
        # acceleration_component = acceleration_component.map(
        #     lambda ac: ac if float(ac) > float(1e-5) else 0.0
        # )
        #
        # self.__positions = (
        #     self.__prev_positions
        #     + dt * self.__prev_velocities
        #     + acceleration_component * (dt * dt * grav_dir)
        # )

        self.__positions = array(self.__inertia)

    def solve(self, dt: float):
        dt_sqr_reciprocal = 1 / dt / dt

        for i in range(1, len(self.__positions)):
            f = (
                self.__vert_mass
                * (self.__inertia[i] - self.__positions[i])
                * dt_sqr_reciprocal
            )
            h = mat3(self.__vert_mass * dt_sqr_reciprocal)

            for edge_id in self.__vert_adjacent_edges[i]:
                v1 = self.__edges[edge_id][0]
                v2 = self.__edges[edge_id][1]
                diff = self.__positions[v1] - self.__positions[v2]
                l = length(diff)
                l0 = self.__lengthes[edge_id]
                # evaluate hessian
                h_1_1 = self.__stiffness * (
                    mat3() - (l0 / l) * (mat3() - tensor_product(diff, diff) / (l * l))
                )
                h += h_1_1

                if v1 == i:
                    f += (self.__stiffness * (l0 - l) / l) * diff

                else:
                    f -= (self.__stiffness * (l0 - l) / l) * diff
                # print("f_p =", f)

            dx = np.linalg.lstsq(np.array(h), np.array(f), rcond=None)[0]
            # print("f=", f)
            self.__positions[i] += vec3(dx)

    def update_velocity(self, dt: float):
        self.__prev_velocities = array(self.__velocities)
        self.__velocities = (self.__positions - self.__prev_positions) / dt
        self.__velocities[0] = vec3(0.0)

    def update(self, ecs: ECS, delta_time: float):
        if delta_time == 0.0:
            return
        dt = delta_time / self.__substep
        for _ in range(self.__substep):
            # print("forward_step")
            self.forward_step(dt)
            # print("solve")
            for _ in range(self.__num_iterations):
                self.solve(dt)
            # print("update_velocity")
            self.update_velocity(dt)
        print(self.__positions)


def setup(ecs: ECS):
    mesh = icosphere(2)

    positions = []
    for i in range(5):
        positions.append(vec3(i * 1.0, 2.5, 0.0))

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
