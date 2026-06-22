from pugtk import Window, Vector3, Matrix4, Camera, Mesh, Renderer3D, color, checker_shader, Texture

window = Window(640, 480)
camera = Camera(
    Vector3(0.0, 0.0, 5.0),
    Vector3(0.0, 0.0, 0.0),
    Vector3(0.0, 1.0, 0.0),
    60.0,
    640.0 / 480.0,
    0.1,
    100.0,
)
renderer = Renderer3D(window, camera)
renderer.pixel_shader = checker_shader
renderer.light_dir = Vector3(0.4, 0.6, 1.0)
renderer.ambient = 0.25
mesh = Mesh.cube(2.0)
no_texture = Texture.solid(1, 1, 0)

# One base color per cube face (back, front, left, right, bottom, top);
# checker_shader darkens alternating 0.5-unit cells within each face.
face_colors = [
    color(255, 0, 0),    # back
    color(0, 255, 0),    # front
    color(0, 0, 255),    # left
    color(255, 255, 0),  # right
    color(255, 0, 255),  # bottom
    color(0, 255, 255),  # top
]

angle = 0.0
running = 1
while running:
    window.clear()
    model = Matrix4.rotation_y(angle).multiply(Matrix4.rotation_x(angle * 0.6))
    renderer.render_solid_per_pixel(mesh, model, face_colors, no_texture)
    running = window.update()
    angle = angle + 0.02

window.close()
