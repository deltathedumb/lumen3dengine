from pugtk import Vector3, Matrix4, Camera, Mesh, color
from pugtk._renderer3d_gl import GLRenderer3D, GLWindow
from pugtk._scene import SceneObject

window = GLWindow("pugtk shadow demo", 800, 600)
camera = Camera(
    Vector3(0.0, 4.0, 9.0),
    Vector3(0.0, 0.0, 0.0),
    Vector3(0.0, 1.0, 0.0),
    60.0,
    800.0 / 600.0,
    0.1,
    100.0,
)
renderer = GLRenderer3D(window, camera)
renderer.light_dir = Vector3(0.4, 0.8, 0.3)
renderer.ambient = 0.2
renderer.shadow_extent = 10.0

ground_mesh = Mesh.plane(16.0)
ground_color = color(120, 120, 130)
ground_colors = [ground_color, ground_color]

cube_mesh = Mesh.cube(2.0)
cube_color = color(220, 60, 60)
cube_colors = [cube_color] * 12

angle = 0.0
running = 1
while running:
    ground_model = Matrix4.identity()
    cube_model = Matrix4.translation(0.0, 2.5, 0.0).multiply(Matrix4.rotation_y(angle))

    ground_obj = SceneObject(ground_mesh, ground_model, ground_colors, None)
    cube_obj = SceneObject(cube_mesh, cube_model, cube_colors, None)

    renderer.render_scene([ground_obj, cube_obj])
    running = renderer.update()
    angle = angle + 0.015

window.close()
