from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Concatenate


@dataclass
class Entity:
    id: int


class ECS:
    def __init__(self):
        self.__components = {Entity: {}}
        self.__next_id = 0
        self.__systems: dict[str, list[System]] = {}

    def spawn(self, components: list):
        id = self.__next_id

        self.__components[Entity][id] = Entity(id)

        for component in components:
            ty = type(component)
            assert ty != Entity, "Entity Components are automatically added"
            if ty not in self.__components:
                self.__components[ty] = {}
            self.__components[ty][id] = component

        self.__next_id += 1
        return id

    def __getitem__(self, id: int):
        res = []
        for ty in self.__components.values():
            if id in ty:
                res.append(ty[id])
        return res

    def query(self, types: list):
        ids = set(self.__components[Entity].keys())
        for ty in types:
            ids = ids & set(self.__components[ty].keys())
        for id in ids:
            yield tuple(self.__components[ty][id] for ty in types)

    def query_one(self, types: list):
        ids = set(self.__components[Entity].keys())
        for ty in types:
            ids = ids & set(self.__components[ty].keys())
        id = ids.pop()
        return tuple(self.__components[ty][id] for ty in types)

    def on(self, event: str, system: System):
        if event not in self.__systems:
            self.__systems[event] = []
        self.__systems[event].append(system)
        return self

    def dispatch(self, event: str, *args, **kwargs):
        if event in self.__systems:
            for system in self.__systems[event]:
                system(self, *args, **kwargs)
        return self

    def do(self, system: System, *args, **kwargs):
        system(self, *args, **kwargs)


System = Callable[Concatenate[ECS, ...], None]

if __name__ == "__main__":
    ecs = ECS()

    ecs.spawn([True, [], 42])
    ecs.spawn([False, 69])
    ecs.spawn([["caca"], 42])

    def print_int(ecs: ECS):
        for (v,) in ecs.query([int]):
            print(v)

    ecs.on("render", print_int)

    ecs.dispatch("render")
