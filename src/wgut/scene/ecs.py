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
    label: str

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"[{self.id}] {self.label}"


class EntityNotFound(Exception):
    def __init__(self, id: int | Entity):
        if isinstance(id, Entity):
            id = id.id
        self.id = id

    def __str__(self):
        return f"Entity {self.id} Not Found"


class QueryOneWithNoResult(Exception):
    def __init__(self, types, without):
        self.types = types
        self.without = without

    def __str__(self):
        return f"Query with={self.types} and without={self.without} has no result"


@dataclass
class Group:
    ids: list[int]


# TODO:
# - Support "Not" in Query
# - Redo Group


class ECS:
    def __init__(self):
        self.__components: dict[Type, dict[int, Any]] = {}
        self.__next_id = 0
        self.__systems: dict[str, list[System]] = {}

    def spawn(self, components: list, label: str | None = None) -> int:
        id = self.__next_id
        self.__next_id += 1
        if label is None:
            label = f"Entity {id}"

        self.__add_component(id, Entity(id, label))

        for component in components:
            assert type(component) is not Entity, (
                "Entity Components are automatically added"
            )
            self.__add_component(id, component)

        return id

    def add_component(self, id: int | Entity, component) -> Self:
        id = self.__entity_exists(id)
        self.__add_component(id, component)
        return self

    def __entity_exists(self, id: int | Entity) -> int:
        if isinstance(id, Entity):
            id = id.id

        if Entity not in self.__components:
            raise EntityNotFound(id)
        if id not in self.__components[Entity]:
            raise EntityNotFound(id)

        return id

    def __add_component(self, id: int, component):
        ty = type(component)
        if ty not in self.__components:
            self.__components[ty] = {}
        self.__components[ty][id] = component

    def remove_component(self, id: int | Entity, ty: Type) -> Self:
        id = self.__entity_exists(id)

        if ty == Entity:
            print("Warning: Cannot remove 'Entity' component")
            return self

        self.__remove_component(id, ty)
        return self

    def __remove_component(self, id: int, ty: Type):
        if ty in self.__components:
            if id in self.__components[ty]:
                self.__components[ty].pop(id)
                if len(self.__components[ty]) == 0:
                    self.__components.pop(ty)

    def __getitem__(self, id: int | Entity) -> dict[Type, Any]:
        id = self.__entity_exists(id)

        res = {}
        for ty, comps in self.__components.items():
            if id in comps:
                res[ty] = comps[id]
        return res

    def kill(self, id: int | Entity) -> Self:
        id = self.__entity_exists(id)

        for ty in list(self.__components):
            self.__remove_component(id, ty)

        return self

    def query(
        self, types: Sequence[Type] | Type, without: Sequence[type] | Type = []
    ) -> Generator[Any, None, None]:
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

        if not isinstance(without, Sequence):
            without = [without]

        for ty in without:
            if ty in self.__components:
                ids = ids - set(self.__components[ty].keys())

        for id in ids:
            res = tuple(self.__components[ty][id] for ty in types)
            if returns_tuple:
                yield res
            else:
                yield res[0]

    def query_one(
        self, types: Sequence[Type] | Type, without: Sequence[type] | Type = []
    ) -> Any:
        for res in self.query(types, without):
            return res
        raise QueryOneWithNoResult(types, without)

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

    print(ecs[0])
    print(ecs.query_one([bool, int]))

    def print_int(ecs: ECS):
        for v in ecs.query([int, bool, Entity]):
            print(v)

    ecs.on("render", print_int)

    ecs.dispatch("render")

    ecs.kill(0)
    ecs.kill(2)

    ecs.dispatch("render")
