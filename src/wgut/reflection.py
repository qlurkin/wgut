import importlib

import wgpu

from wgut.builders import BingGroupLayoutBuilder, PipelineLayoutBuilder

tswgsl = importlib.import_module("tree_sitter_wgsl_bevy")
ts = importlib.import_module("tree_sitter")

WGSL_LANGUAGE = ts.Language(tswgsl.language())
parser = ts.Parser(WGSL_LANGUAGE)


class Reflection:
    def __init__(self, source):
        self.source = source
        self.groups = {}
        self.pipeline_layout = None
        self._parse()

    def query(self, query_str):
        query = WGSL_LANGUAGE.query(query_str)
        return query.matches(self.root)

    def _parse(self):
        tree = parser.parse(self.source.encode())
        self.root = tree.root_node
        # print(self.root)

        matches = self.query(
            "(global_variable_declaration (attribute (identifier) @attr.type (int_literal) @attr.id)* (variable_declaration (variable_qualifier (address_space) @space (access_mode)? @access)? (variable_identifier_declaration name: (identifier) @id type: (type_declaration) @type)))"
        )

        for match in matches:
            match = match[1]
            # print("MATCH GLOBAL:", match)
            attributes = {}
            for ty, id in zip(match["attr.type"], match["attr.id"]):
                attributes[ty.text.decode()] = int(id.text.decode())
            access = None
            space = None
            if "access" in match:
                access = match["access"][0].text.decode()
            if "space" in match:
                space = match["space"][0].text.decode()
            if "group" in attributes:
                if attributes["group"] not in self.groups:
                    self.groups[attributes["group"]] = {}
                self.groups[attributes["group"]][attributes["binding"]] = {
                    "name": match["id"][0].text.decode(),
                    "type": match["type"][0].text.decode(),
                    "access": access,
                    "space": space,
                }

        matches = self.query(
            "(function_declaration (attribute)* @attributes name: (identifier) @name parameters: (parameter_list) body: (compound_statement) @body)"
        )
        functions = []
        for match in matches:
            match = match[1]
            attributes = {}
            for attr in match["attributes"]:
                typ = None
                values = []
                for child in attr.children:
                    if child.type == "identifier":
                        typ = child.text.decode()
                    if child.type == "int_literal":
                        values.append(int(child.text.decode()))
                attributes[typ] = values

            # for ty, id in zip(match["attr.type"], match["attr.id"]):
            # attributes[ty.text.decode()] = int(id.text.decode())
            functions.append(
                {
                    "name": match["name"][0].text.decode(),
                    "attributes": attributes,
                    "body": match["body"][0].text.decode(),
                    "body_tree": match["body"][0],
                }
            )
        self.entry_points = {"compute": [], "vertex": [], "fragment": []}
        for fun in functions:
            for stage in self.entry_points:
                if stage in fun["attributes"]:
                    self.entry_points[stage].append(fun["name"])
        self.workgroup_size = {}
        for fun in functions:
            if "workgroup_size" in fun["attributes"]:
                size = fun["attributes"]["workgroup_size"]
                while len(size) < 3:
                    size.append(1)
                self.workgroup_size[fun["name"]] = size

        for group in self.groups.values():
            for binding in group.values():
                visibility = self._find_visibility(binding["name"], functions)
                binding["visibility"] = visibility

        for index in self.groups:
            self.groups[index]["layout"] = self._create_bind_group_layout(index)

        self.pipeline_layout = self._create_pipeline_layout()

    def entry_point_status(self, name):
        res = set()
        for entry in self.entry_points:
            if name in self.entry_points[entry]:
                res.add(entry)
        return res

    def _find_visibility(self, name, all_functions):
        names = set([name])
        done = set()
        res = set()
        functions = set()
        while len(names) > 0:
            name = names.pop()
            done.add(name)
            for fun in all_functions:
                q = WGSL_LANGUAGE.query(f'((identifier) @name (#eq? @name "{name}"))')
                matches = q.matches(fun["body_tree"])
                if len(matches) > 0:
                    functions.add(fun["name"])
                    status = self.entry_point_status(fun["name"])
                    if len(status) > 0:
                        res = set(list(res) + list(status))
                    else:
                        if name not in done:
                            names.add(fun["name"])
        return res

    def _create_bind_group_layout(self, index):
        VISIBILITIES = {
            "fragment": wgpu.ShaderStage.FRAGMENT,
            "vertex": wgpu.ShaderStage.VERTEX,
            "compute": wgpu.ShaderStage.COMPUTE,
        }
        builder = BingGroupLayoutBuilder()
        for binding_index in sorted(list(self.groups[index].keys())):
            binding = self.groups[index][binding_index]
            visibility = 0
            for v in binding["visibility"]:
                visibility = visibility | VISIBILITIES[v]
            if binding["type"].startswith("texture"):
                builder.with_texture(visibility, binding_index)
            elif binding["type"] == "sampler":
                builder.with_sampler(visibility, binding_index)
            else:
                bufferType = "uniform"
                if binding["space"] == "storage":
                    if binding["access"] is None or binding["access"] == "read":
                        bufferType = "read-only-storage"
                    else:
                        bufferType = "storage"
                builder.with_buffer(visibility, bufferType, binding_index)
        return builder.build()

    def _create_pipeline_layout(self):
        builder = PipelineLayoutBuilder()
        for index in sorted(list(self.groups.keys())):
            builder.with_bind_group_layout(self.groups[index]["layout"])
        return builder.build()

    def get_bind_group_layout(self, index):
        return self.groups[index]["layout"]
