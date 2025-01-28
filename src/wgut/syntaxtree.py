import importlib

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
