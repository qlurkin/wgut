from wgut import (
    ShaderToy,
    create_canvas,
    load_file,
)

canvas = create_canvas()

ShaderToy(canvas, load_file("./sea.wgsl")).run()

