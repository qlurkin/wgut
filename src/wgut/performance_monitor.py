from imgui_bundle import imgui, implot
from wgut.ecs import ECS
import numpy as np

from wgut.window import Window

frame_times = []


def performance_monitor(ecs: ECS):
    frame_times = [0.0] * 100
    render_times = [0.0] * 100
    update_times = [0.0] * 100
    implot.create_context()

    def setup(ecs: ECS, window: Window):
        def gui(ecs: ECS):
            render_times.append(window.last_render_time)
            render_times.pop(0)
            frame_times.append(window.last_frame_time)
            frame_times.pop(0)
            update_times.append(window.last_update_time)
            update_times.pop(0)
            imgui.begin("Performance Monitor", None)
            if imgui.collapsing_header("Frame time"):
                imgui.text(f"Frame Time: {frame_times[-1]:.5f}s")
                if implot.begin_plot("##Frame time", (-1, 100)):
                    implot.setup_axes("##f", "##ft", implot.AxisFlags_.auto_fit.value)
                    implot.setup_axes_limits(-100, 0, 0, 0.025)
                    implot.plot_line(
                        "##ft(f)",
                        np.linspace(-100, 0, 100, dtype=np.float64),
                        np.array(frame_times),
                    )
                    implot.end_plot()
            if imgui.collapsing_header("Update time"):
                imgui.text(f"Render Time: {update_times[-1]:.5f}s")
                if implot.begin_plot("##Update time", (-1, 100)):
                    implot.setup_axes("##f", "##rt", implot.AxisFlags_.auto_fit.value)
                    implot.setup_axes_limits(-100, 0, 0, 0.025)
                    implot.plot_line(
                        "##rt(f)",
                        np.linspace(-100, 0, 100, dtype=np.float64),
                        np.array(update_times),
                    )
                    implot.end_plot()
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

            imgui.end()

        ecs.on("render_gui", gui)

    ecs.on("setup", setup)
