from imgui_bundle import imgui, implot
from wgut.scene.ecs import ECS, EntityNotFound
from wgut.scene.render_system3 import RenderStat
import numpy as np

frame_times = []


def performance_monitor(ecs: ECS):
    frame_times = [0.0]
    render_times = {}
    implot.create_context()

    def update(ecs: ECS, delta_time: float):
        frame_times.append(delta_time)
        if len(frame_times) > 100:
            frame_times.pop(0)

    def gui(ecs: ECS):
        try:
            r_stat: RenderStat = ecs.query_one(RenderStat)
            layers = list(r_stat.stats.keys())

            times = frame_times
            if len(times) < 100:
                times = [0] * (100 - len(times)) + times
            imgui.begin("Performance Monitor", None)
            if imgui.collapsing_header("Frame time"):
                imgui.text(f"Frame Time: {frame_times[-1]:.5f}s")
                if implot.begin_plot("##Frame time", (-1, 100)):
                    implot.setup_axes("##f", "##ft", implot.AxisFlags_.auto_fit.value)
                    implot.setup_axes_limits(-100, 0, 0, 0.025)
                    implot.plot_line(
                        "##ft(f)",
                        np.linspace(-100, 0, 100, dtype=np.float64),
                        np.array(times),
                    )
                    implot.end_plot()
            if imgui.begin_tab_bar("layers"):
                for layer in layers:
                    if layer not in render_times:
                        render_times[layer] = [0] * 100

                    stat = r_stat.stats[layer]
                    if imgui.begin_tab_item(layer.name):
                        render_times[layer].append(stat["time"])
                        render_times[layer].pop(0)
                        if imgui.collapsing_header("Render time"):
                            imgui.text(f"Render Time: {render_times[layer][-1]:.5f}s")
                            if implot.begin_plot(
                                f"##Render time {layer.name}", (-1, 100)
                            ):
                                implot.setup_axes(
                                    "##f", "##rt", implot.AxisFlags_.auto_fit.value
                                )
                                implot.setup_axes_limits(-100, 0, 0, 0.025)
                                implot.plot_line(
                                    "##rt(f)",
                                    np.linspace(-100, 0, 100, dtype=np.float64),
                                    np.array(render_times[layer]),
                                )
                                implot.end_plot()
                        imgui.text(f"Draw count: {stat['draw']}")
                        imgui.text(f"Mesh count: {stat['mesh']}")
                        imgui.text(f"Triangle count: {stat['triangle']}")
                        imgui.text(f"Vertex count: {stat['vertex']}")
                        imgui.end_tab_item()
                imgui.end_tab_bar()
            imgui.end()
        except EntityNotFound:
            pass

    ecs.on("update", update)
    ecs.on("render_gui", gui)
