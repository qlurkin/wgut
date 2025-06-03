from __future__ import annotations


class BuilderBase:
    def __init__(self, label: str = "No Label"):
        self.label = label

    def with_label(self, label):
        self.label = label
        return self
