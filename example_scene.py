from pugtk import Window, Vector3, Matrix4, Camera, Mesh, Renderer3D, color, lambert_shader, Texture, SceneObject

window = Window(640, 480)
camera = Camera(
    Vector3(0.0, 1.0, 8.0),
    Vector3(0.0, 0.0, 0.0),
    Vector3(0.0, 1.0, 0.0),
    60.0,
    640.0 / 480.0,
    0.1,
    100.0,
)
renderer = Renderer3D(window, camera)
renderer.pixel_shader = lambert_shader
renderer.light_dir = Vector3(0.4, 0.6, 1.0)
renderer.ambient = 0.25
no_texture = Texture.solid(1, 1, 0)

mesh_a = Mesh.cube(1.6)
mesh_b = Mesh.cube(1.2)
mesh_c = Mesh.cube(2.0)

red = color(220, 60, 60)
blue = color(60, 100, 220)
yellow = color(220, 200, 60)
colors_a = [red, red, red, red, red, red]
colors_b = [blue, blue, blue, blue, blue, blue]
colors_c = [yellow, yellow, yellow, yellow, yellow, yellow]

angle = 0.0
running = 1
while running:
    window.clear()

    # Three cubes orbiting at different radii/speeds/depths -- each
    # frame's positions deliberately overlap on screen at times, so
    # render_scene()'s cross-object depth sort actually has to do work
    # (not just three cubes that happen to never occlude each other).
    model_a = Matrix4.translation(-1.5, 0.0, 1.5).multiply(Matrix4.rotation_y(angle))
    model_b = Matrix4.translation(1.5, 0.0, 0.0).multiply(Matrix4.rotation_y(angle * 1.3))
    model_c = Matrix4.translation(0.0, 0.0, -1.5).multiply(Matrix4.rotation_y(angle * 0.7))

    objects = [
        SceneObject(mesh_a, model_a, colors_a, no_texture),
        SceneObject(mesh_b, model_b, colors_b, no_texture),
        SceneObject(mesh_c, model_c, colors_c, no_texture),
    ]
    renderer.render_scene(objects)

    running = window.update()
    angle = angle + 0.02

window.close()
