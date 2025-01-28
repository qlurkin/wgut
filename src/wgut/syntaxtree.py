import importlib
from typing import Self

tswgsl = importlib.import_module("tree_sitter_wgsl_bevy")
ts = importlib.import_module("tree_sitter")

WGSL_LANGUAGE = ts.Language(tswgsl.language())


class SyntaxTree:
    def __init__(self, source):
        parser = ts.Parser(WGSL_LANGUAGE)
        tree = parser.parse(source.encode())
        self.__root = tree.root_node
        self.__source = source

    def get_source(self):
        return self.__source

    def query(self, query_str, root=None):
        if root is None:
            root = self.__root
        query = WGSL_LANGUAGE.query(query_str)
        return query.matches(root)


class Stack:
    def __init__(self):
        self.__data = []

    def push(self, *args):
        self.__data.append(args)

    def pop(self):
        return self.__data.pop()

    def is_empty(self):
        return len(self.__data) == 0


class SyntaxNode:
    def __init__(self, rule_name: str, children: list[Self | str]):
        self.rule_name = rule_name
        self.children = children

    def __repr__(self):
        parts = [self.rule_name]
        for child in self.children:
            if isinstance(child, str):
                parts.append(f"'{child}'")
            else:
                parts.append(str(child))
        return f"({' '.join(parts)})"


class Terminal:
    def __init__(self, txt: str):
        self.txt = txt


class Parser:
    def __init__(self, rules):
        self.source = ""
        self.cursor = 0
        self.rules = {}
        for predicate, expansion in rules.items():
            if not isinstance(expansion, Choice):
                self.rules[predicate] = Choice(expansion).expand()
            else:
                self.rules[predicate] = expansion.expand()

        print(self.rules)

    def use_rule(self, rule_name: str, cursor: int) -> tuple[SyntaxNode | None, int]:
        rule = self.rules[rule_name]
        for option in rule:
            children = []
            cur = cursor
            option_ok = True
            for item in option.items():
                if isinstance(item, NonTerm):
                    child, cur = self.use_rule(str(item), cur)
                    if child is not None:
                        children.append(child)
                    else:
                        option_ok = False
                        break
                elif isinstance(item, str):
                    ok = True
                    for c in item:
                        if cur < len(self.source) and self.source[cur] == c:
                            cur += 1
                        else:
                            ok = False
                            break
                    if ok:
                        children.append(item)
                    else:
                        option_ok = False
                        break
            if option_ok:
                return SyntaxNode(rule_name, children), cur
        return None, cursor

    def parse(self, source):
        self.source = source
        node, cur = self.use_rule("start", 0)
        if cur == len(source):
            return node
        return None


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


HaaGrammar = {
    "start": Seq("h", NonTerm("tail")),
    "tail": Choice(Seq("a", NonTerm("tail")), "a"),
}

RegexGrammar = {
    "start": Optional(NonTerm("expression")),
    "expression": Choice(Seq(NonTerm("term"), NonTerm("expression")), NonTerm("term")),
    "term": Choice(Seq(NonTerm("item"), NonTerm("modifier")), NonTerm("item")),
    "modifier": Choice("+", "*"),
    "item": Choice(NonTerm("character"), NonTerm("group")),
    "group": Seq("(", NonTerm("expression"), ")"),
    "character": Choice("a", "b", "c"),
}


if __name__ == "__main__":
    print(HaaGrammar)
    P = Parser(HaaGrammar)
    print(P.parse("haaa"))

    P = Parser(RegexGrammar)
    print(P.parse("(ca+)*"))
