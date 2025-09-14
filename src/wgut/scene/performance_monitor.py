from imgui_bundle import imgui, implot
from wgut.scene.ecs import ECS, EntityNotFound
from wgut.scene.render_system import RenderStats
import numpy as np

frame_times = []


def performance_monitor(ecs: ECS):
    frame_times = [0.0]
    render_times = [0] * 100
    implot.create_context()

    def update(ecs: ECS, delta_time: float):
        frame_times.append(delta_time)
        if len(frame_times) > 100:
            frame_times.pop(0)

    def gui(ecs: ECS):
        try:
            r_stat: RenderStats = ecs.query_one(RenderStats)

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
            stats = r_stat.stats
            render_times.append(stats["time"])
            render_times.pop(0)
            if imgui.collapsing_header("Render time"):
                imgui.text(f"Render Time: {render_times[-1]:.5f}s")
                if implot.begin_plot("##Render time", (-1, 100)):
                    implot.setup_axes("##f", "##rt", implot.AxisFlags_.auto_fit.value)
                    implot.setup_axes_limits(-100, 0, 0, 0.025)
                    implot.plot_line(
                        "##rt(f)",
                        np.linspace(-100, 0, 100, dtype=np.float64),
                        np.array(render_times),
                    )
                    implot.end_plot()
            imgui.text(f"Draw count: {stats['draw']}")
            imgui.text(f"Mesh count: {stats['mesh']}")
            imgui.text(f"Triangle count: {stats['triangle']}")
            imgui.text(f"Vertex count: {stats['vertex']}")
            imgui.end()
        except EntityNotFound:
            pass

    ecs.on("update", update)
    ecs.on("render_gui", gui)
