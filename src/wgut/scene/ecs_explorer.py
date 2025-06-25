from imgui_bundle import imgui
from wgut.scene.ecs import ECS, Entity

frame_times = []


def ecs_explorer(ecs: ECS):
    def gui(ecs: ECS):
        imgui.begin("ECS Explorer", None)
        for (entity,) in ecs.query([Entity]):
            if imgui.tree_node(str(entity)):
                components = ecs[entity]
                for component in components:
                    if isinstance(component, Entity):
                        continue
                    if imgui.tree_node(str(component)):
                        if "ecs_explorer_gui" in dir(component):
                            component.ecs_explorer_gui()  # type: ignore
                        imgui.tree_pop()
                imgui.tree_pop()
        imgui.end()

    ecs.on("render_gui", gui)
