from __future__ import annotations

import inspect


class BuilderBase:
    def __init__(self):
        name = type(self).__name__
        stack = inspect.stack()
        module_name = "wgut."
        module_path = "some_path"
        line = -1
        depth = 0
        while module_name.startswith("wgut."):
            parent_frame = stack[depth][0]
            module_info = inspect.getmodule(parent_frame)
            if module_info is None:
                break
            module_name = module_info.__name__
            module_path = module_info.__file__
            line = parent_frame.f_lineno
            depth += 1
        self.label = f"Made by {name} in {module_path} at line {line}"

    def with_label(self, label):
        self.label = label
        return self
