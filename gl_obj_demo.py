from pugtk import Vector3, Matrix4, Camera, color, load_obj
from pugtk._renderer3d_gl import GLRenderer3D, GLWindow

window = GLWindow("lumen3dengine OBJ loader demo", 640, 480)
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
renderer.light_dir = Vector3(0.4, 0.6, 1.0)
renderer.ambient = 0.25

mesh = load_obj("test_cube.obj")
mesh_color = color(80, 160, 230)
colors = [mesh_color] * len(mesh.triangles)

angle = 0.0
running = 1
while running:
    model = Matrix4.rotation_y(angle).multiply(Matrix4.rotation_x(angle * 0.6))
    renderer.render_solid(mesh, model, colors)
    running = renderer.update()
    angle = angle + 0.02

window.close()
