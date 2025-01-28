from typing import Self
from grammar import Choice, NonTerm, Seq, Optional, Repeat


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
    def __init__(
        self,
        rule_name: str,
        children: list[Self | str],
        start: int,
        end: int,
        source: str,
    ):
        self.rule_name = rule_name
        self.children = children
        self.start = start
        self.end = end
        self.source = source

    def __repr__(self):
        parts = [self.rule_name]
        for child in self.children:
            if isinstance(child, str):
                parts.append(f"'{child}'")
            else:
                parts.append(str(child))
        return f"({' '.join(parts)})"

    def search(self, rule_name):
        res = []
        if self.rule_name == rule_name:
            res.append(self)
        for child in self.children:
            if isinstance(child, SyntaxNode):
                res += child.search(rule_name)
        return res

    @property
    def text(self):
        return self.source[self.start : self.end]


def cleanup(node: SyntaxNode) -> SyntaxNode:
    new_children = []
    for child in node.children:
        if isinstance(child, SyntaxNode):
            child = cleanup(child)
            if child.rule_name.startswith("_"):
                new_children = new_children + child.children
            else:
                new_children.append(child)
        else:
            new_children.append(child)
    return SyntaxNode(node.rule_name, new_children, node.start, node.end, node.source)


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

        for expansion in self.rules.values():
            for i, option in enumerate(expansion):
                expansion[i] = option.flatten()

        repeat_id = 0
        repeat_rules = {}
        for expansion in self.rules.values():
            for i, option in enumerate(expansion):
                res = []
                for item in option.items():
                    if isinstance(item, Repeat):
                        rule_name = f"_repeat_{repeat_id}"
                        repeat_id += 1
                        res.append(NonTerm(rule_name))
                        repeat_rules[rule_name] = Choice(
                            Seq(item.item(), NonTerm(rule_name)).flatten(),
                            Seq(item.item()).flatten(),
                        ).expand()
                    else:
                        res.append(item)
                expansion[i] = Seq(*res)

        self.rules.update(repeat_rules)

        print(self.rules)

    def use_rule(self, rule_name: str, cursor: int) -> SyntaxNode | None:
        rule = self.rules[rule_name]
        for option in rule:
            children = []
            cur = cursor
            option_ok = True
            for item in option.items():
                if isinstance(item, NonTerm):
                    child = self.use_rule(str(item), cur)
                    if child is not None:
                        cur = child.end
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
                return SyntaxNode(rule_name, children, cursor, cur, self.source)
        return None

    def parse(self, source):
        self.source = source
        node = self.use_rule("start", 0)
        if node is not None:
            if node.end == len(source):
                return cleanup(node)
        return None


if __name__ == "__main__":
    HaaGrammar = {
        "start": Seq("h", NonTerm("tail")),
        # "start": Seq("h", Repeat("a")),
        # "tail": Choice(Seq("a", NonTerm("tail")), "a"),
        "tail": Repeat("a"),
    }

    RegexGrammar = {
        "start": Optional(NonTerm("expression")),
        "expression": Choice(
            Seq(NonTerm("term"), NonTerm("expression")), NonTerm("term")
        ),
        "term": Choice(Seq(NonTerm("item"), NonTerm("modifier")), NonTerm("item")),
        "modifier": Choice("+", "*"),
        "item": Choice(NonTerm("character"), NonTerm("group")),
        "group": Seq("(", NonTerm("expression"), ")"),
        "character": Choice("a", "b", "c"),
    }

    print(HaaGrammar)
    P = Parser(HaaGrammar)
    print(P.parse("haaa"))

    P = Parser(RegexGrammar)
    tree = P.parse("(ca+)*")
    print(tree)
    if tree is not None:
        for node in tree.search("term"):
            print(node.text)
