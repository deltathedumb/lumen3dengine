from pugtk import Vector3, Matrix4, Camera, Mesh, color, Texture
from pugtk._renderer3d_gl import GLRenderer3D, GLWindow
from pugtk._scene import SceneObject

window = GLWindow("pugtk texture demo", 640, 480)
camera = Camera(
    Vector3(0.0, 0.0, 5.0),
    Vector3(0.0, 0.0, 0.0),
    Vector3(0.0, 1.0, 0.0),
    60.0,
    640.0 / 480.0,
    0.1,
    100.0,
)
renderer = GLRenderer3D(window, camera)
renderer.shadows_enabled = 0
renderer.light_dir = Vector3(0.4, 0.6, 1.0)
renderer.ambient = 0.35

# 8x8 checkerboard texture, orange/white squares
size = 8
pixels: list[int] = []
y = 0
while y < size:
    x = 0
    while x < size:
        if (x + y) % 2 == 0:
            pixels.append(color(255, 140, 0))
        else:
            pixels.append(color(255, 255, 255))
        x = x + 1
    y = y + 1
checker_tex = Texture(size, size, pixels)

mesh = Mesh.cube(2.5)
colors = [color(255, 255, 255)] * 12

angle = 0.0
running = 1
while running:
    model = Matrix4.rotation_y(angle).multiply(Matrix4.rotation_x(angle * 0.5))
    obj = SceneObject(mesh, model, colors, checker_tex)
    renderer.render_scene([obj])
    running = renderer.update()
    angle = angle + 0.015

window.close()
