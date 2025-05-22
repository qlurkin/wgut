from __future__ import annotations

from dataclasses import dataclass
from itertools import product
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


# TODO:
# - Support "Not" in Query => Class "Not"
# - Support Multiplicity in Query => Class "With" avec multiplicite a 1 par defaut
#           => If class not "Not" nor "With" wrap with a "With"


class ECS:
    def __init__(self):
        self.__components: dict[Type, dict[int, list[Any]]] = {}
        self.__next_id = 0
        self.__systems: dict[str, list[System]] = {}

    def spawn(self, components: list) -> Self:
        id = self.__next_id

        if Entity not in self.__components:
            self.__components[Entity] = {}
        self.__components[Entity][id] = [Entity(id)]

        for component in components:
            ty = type(component)
            assert ty != Entity, "Entity Components are automatically added"
            if ty not in self.__components:
                self.__components[ty] = {}
            if id not in self.__components[ty]:
                self.__components[ty][id] = []
            self.__components[ty][id].append(component)

        self.__next_id += 1
        return self

    def add_component(self, id: int | Entity, component) -> Self:
        if isinstance(id, Entity):
            id = Entity.id

        if Entity not in self.__components:
            raise EntityNotFound()
        if id not in self.__components[Entity]:
            raise EntityNotFound()
        ty = type(component)
        if ty not in self.__components:
            self.__components[ty] = {}
        if id not in self.__components[ty]:
            self.__components[ty][id] = []
        self.__components[ty][id].append(component)
        return self

    def remove_component(self, id: int | Entity, component) -> Self:
        if isinstance(id, Entity):
            id = Entity.id

        if Entity not in self.__components:
            raise EntityNotFound()
        if id not in self.__components[Entity]:
            raise EntityNotFound()
        ty = type(component)
        if ty in self.__components:
            if id in self.__components[ty]:
                if component in self.__components[ty][id]:
                    self.__components[ty][id].remove(component)
                    if len(self.__components[ty][id]) == 0:
                        self.__components[ty].pop(id)
                    if len(self.__components[ty]) == 0:
                        self.__components.pop(ty)
        return self

    def __getitem__(self, id: int | Entity) -> tuple[list[Any], ...]:
        if isinstance(id, Entity):
            id = Entity.id

        res = []
        for ty in self.__components.values():
            if id in ty:
                res.append(ty[id])
        return tuple(res)

    def kill(self, id: int | Entity) -> Self:
        if isinstance(id, Entity):
            id = Entity.id

        for ty in list(self.__components):
            if id in self.__components[ty]:
                self.__components[ty].pop(id)
            if len(self.__components[ty]) == 0:
                self.__components.pop(ty)
        return self

    def query(self, types: Sequence[Type] | Type) -> Generator[Any, None, None]:
        returns_tuple = False
        if isinstance(types, Sequence):
            returns_tuple = True
        else:
            returns_tuple = False
            types = [types]

        ids = set(self.__components[Entity].keys())
        for ty in types:
            if ty in self.__components:
                ids = ids & set(self.__components[ty].keys())
            else:
                ids = set()
                break

        for id in ids:
            comps = tuple(self.__components[ty][id] for ty in types)
            for res in product(*comps):
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
        res = tuple(self.__components[ty][id][0] for ty in types)
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

    ecs.spawn([True, True, [], 42, 99])
    ecs.spawn([False, 69])
    ecs.spawn([["caca"], 42])

    print(ecs[0])
    print(ecs.query_one([bool, int]))

    def print_int(ecs: ECS):
        for v in ecs.query([int, bool]):
            print(v)

    ecs.on("render", print_int)

    ecs.dispatch("render")

    ecs.kill(0)
    ecs.kill(2)
