from pugtk import Vector3, Matrix4, Camera, Mesh, color, PointLight
from pugtk._renderer3d_gl import GLRenderer3D, GLWindow
from pugtk._scene import SceneObject

window = GLWindow("pugtk multi-light demo", 800, 600)
camera = Camera(
    Vector3(0.0, 3.0, 9.0),
    Vector3(0.0, 0.0, 0.0),
    Vector3(0.0, 1.0, 0.0),
    60.0,
    800.0 / 600.0,
    0.1,
    100.0,
)
renderer = GLRenderer3D(window, camera)
renderer.light_dir = Vector3(0.2, 0.9, 0.1)
renderer.ambient = 0.08
renderer.shadow_extent = 10.0

ground_mesh = Mesh.plane(16.0)
ground_color = color(90, 90, 100)
ground_colors = [ground_color, ground_color]

cube_mesh = Mesh.cube(1.4)
cube_color = color(230, 230, 230)
cube_colors = [cube_color] * 12

red_light = PointLight.omni(Vector3(-3.0, 1.5, 0.0), Vector3(1.0, 0.15, 0.15), 6.0)
blue_light = PointLight.omni(Vector3(3.0, 1.5, 0.0), Vector3(0.15, 0.3, 1.0), 6.0)
green_spot = PointLight(
    Vector3(0.0, 4.0, 2.0),
    Vector3(0.2, 1.0, 0.3),
    10.0,
    Vector3(0.0, -1.0, -0.3),
    25.0,
)
renderer.point_lights = [red_light, blue_light, green_spot]

angle = 0.0
running = 1
while running:
    ground_model = Matrix4.identity()
    cube_model = Matrix4.translation(0.0, 0.8, 0.0).multiply(Matrix4.rotation_y(angle))

    ground_obj = SceneObject(ground_mesh, ground_model, ground_colors, None)
    cube_obj = SceneObject(cube_mesh, cube_model, cube_colors, None)

    renderer.render_scene([ground_obj, cube_obj])
    running = renderer.update()
    angle = angle + 0.01

window.close()
