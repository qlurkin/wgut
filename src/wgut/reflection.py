import wgpu

from wgut.builders import BingGroupLayoutBuilder, PipelineLayoutBuilder
from wgut.syntaxtree import SyntaxTree


class Reflection:
    def __init__(self, source):
        self.__groups = {}
        self.__tree = SyntaxTree(source)
        self.__parse()

    def __parse(self):
        matches = self.__tree.query(
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
                if attributes["group"] not in self.__groups:
                    self.__groups[attributes["group"]] = {}
                self.__groups[attributes["group"]][attributes["binding"]] = {
                    "name": match["id"][0].text.decode(),
                    "type": match["type"][0].text.decode(),
                    "access": access,
                    "space": space,
                }

        matches = self.__tree.query(
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

        for group in self.__groups.values():
            for binding in group.values():
                visibility = self.__find_visibility(binding["name"], functions)
                binding["visibility"] = visibility

        for index in self.__groups:
            self.__groups[index]["layout"] = self.__create_bind_group_layout(index)

        self.__pipeline_layout = self.__create_pipeline_layout()

    def get_bind_group_ids(self) -> list[int]:
        return list(self.__groups.keys())

    def get_binding_ids(self, group_id) -> list[int]:
        res = list(self.__groups[group_id].keys())
        res.remove("layout")
        return res

    def get_binding_space(self, group_id, binding_id):
        return self.__groups[group_id][binding_id]["space"]

    def __entry_point_status(self, name):
        res = set()
        for entry in self.entry_points:
            if name in self.entry_points[entry]:
                res.add(entry)
        return res

    def __find_visibility(self, name, all_functions):
        names = set([name])
        done = set()
        res = set()
        functions = set()
        while len(names) > 0:
            name = names.pop()
            done.add(name)
            for fun in all_functions:
                matches = self.__tree.query(
                    f'((identifier) @name (#eq? @name "{name}"))', fun["body_tree"]
                )
                if len(matches) > 0:
                    functions.add(fun["name"])
                    status = self.__entry_point_status(fun["name"])
                    if len(status) > 0:
                        res = set(list(res) + list(status))
                    else:
                        if name not in done:
                            names.add(fun["name"])
        return res

    def __create_bind_group_layout(self, index):
        VISIBILITIES = {
            "fragment": wgpu.ShaderStage.FRAGMENT,
            "vertex": wgpu.ShaderStage.VERTEX,
            "compute": wgpu.ShaderStage.COMPUTE,
        }
        builder = BingGroupLayoutBuilder()
        for binding_index in sorted(list(self.__groups[index].keys())):
            binding = self.__groups[index][binding_index]
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

    def __create_pipeline_layout(self):
        builder = PipelineLayoutBuilder()
        for index in sorted(list(self.__groups.keys())):
            builder.with_bind_group_layout(self.__groups[index]["layout"])
        return builder.build()

    def get_bind_group_layout(self, index):
        return self.__groups[index]["layout"]

    def get_pipeline_layout(self):
        return self.__pipeline_layout

    def get_source(self):
        return self.__tree.get_source()
