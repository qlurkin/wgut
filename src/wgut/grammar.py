import re


class NonTerm:
    def __init__(self, rule_name: str):
        self.__rule_name = rule_name

    def __str__(self):
        return self.__rule_name

    def __repr__(self):
        return f"NonTerm({self.__rule_name})"


class Seq:
    def __init__(self, *args):
        self.__items = args

    def items(self):
        return self.__items

    def flatten(self):
        res = tuple()
        for item in self.__items:
            if isinstance(item, Seq):
                res = res + item.items()
            else:
                res = res + (item,)
        return Seq(*res)

    def __add__(self, other):
        items = self.__items + other.__items
        return Seq(*items)

    def __repr__(self):
        return "Seq" + str(self.__items)


class Choice:
    def __init__(self, *args):
        def ensure_seq(item):
            if not isinstance(item, Seq):
                return Seq(item)
            return item

        self.options = tuple(map(ensure_seq, args))

    def expand(self) -> list[Seq]:
        res = []
        for option in self.options:
            expanded = [Seq()]
            for item in option.items():
                if isinstance(item, Choice):
                    branches = item.expand()
                    new_expanded = []
                    for e in expanded:
                        for b in branches:
                            new_expanded.append(e + b)
                    expanded = new_expanded
                else:
                    for i in range(len(expanded)):
                        expanded[i] = expanded[i] + Seq(item)
            res = res + expanded
        return res

    def __repr__(self):
        return "Choice" + str(self.options)


class Optional(Choice):
    def __init__(self, item):
        super().__init__(item, "")


class Repeat:
    def __init__(self, item):
        self.__item = item

    def item(self):
        return self.__item

    def __repr__(self):
        if isinstance(self.__item, str):
            content = f"'{self.__item}'"
        else:
            content = str(self.__item)
        return f"Repeat({content})"


class Regex:
    def __init__(self, pattern):
        self.pattern = re.compile(pattern)
