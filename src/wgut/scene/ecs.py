from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Concatenate,
    Generator,
    Self,
    Sequence,
    Type,
)


@dataclass
class Entity:
    id: int


class EntityNotFound(Exception):
    pass


class ECS:
    def __init__(self):
        self.__components = {Entity: {}}
        self.__next_id = 0
        self.__systems: dict[str, list[System]] = {}

    def spawn(self, components: list) -> Self:
        id = self.__next_id

        self.__components[Entity][id] = Entity(id)

        for component in components:
            ty = type(component)
            assert ty != Entity, "Entity Components are automatically added"
            if ty not in self.__components:
                self.__components[ty] = {}
            self.__components[ty][id] = component

        self.__next_id += 1
        return self

    def __getitem__(self, id: int):
        res = []
        for ty in self.__components.values():
            if id in ty:
                res.append(ty[id])
        return res

    def query(self, types: Sequence[Type] | Type) -> Generator[Any, None, None]:
        returns_tuple = False
        if isinstance(types, Sequence):
            returns_tuple = True
        else:
            returns_tuple = False
            types = [types]

        ids = set(self.__components[Entity].keys())
        for ty in types:
            ids = ids & set(self.__components[ty].keys())
        for id in ids:
            res = tuple(self.__components[ty][id] for ty in types)
            if returns_tuple:
                yield res
            else:
                yield res[0]

    def query_one(self, types: Sequence[Type] | Type) -> Any:
        returns_tuple = False
        if isinstance(types, Sequence):
            returns_tuple = True
        else:
            returns_tuple = False
            types = [types]

        ids = set(self.__components[Entity].keys())
        for ty in types:
            if ty not in self.__components:
                raise EntityNotFound()
            ids = ids & set(self.__components[ty].keys())
        if len(ids) == 0:
            raise EntityNotFound()
        id = ids.pop()
        res = tuple(self.__components[ty][id] for ty in types)
        if returns_tuple:
            return res
        else:
            return res[0]

    def on(self, event: str, system: System) -> Self:
        if event not in self.__systems:
            self.__systems[event] = []
        self.__systems[event].append(system)
        return self

    def dispatch(self, event: str, *args, **kwargs) -> Self:
        if event in self.__systems:
            for system in self.__systems[event]:
                system(self, *args, **kwargs)
        return self

    def do(self, system: System, *args, **kwargs) -> Self:
        system(self, *args, **kwargs)
        return self


System = Callable[Concatenate[ECS, ...], None]

if __name__ == "__main__":
    ecs = ECS()

    ecs.spawn([True, [], 42])
    ecs.spawn([False, 69])
    ecs.spawn([["caca"], 42])

    def print_int(ecs: ECS):
        for v in ecs.query(int):
            print(v)

    ecs.on("render", print_int)

    ecs.dispatch("render")
