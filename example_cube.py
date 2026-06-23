from pugtk import Window, Vector3, Matrix4, Camera, Mesh, Renderer3D, color, lambert_shader

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
renderer.shader = lambert_shader
mesh = Mesh.cube(2.0)

# Two colors per cube face (back, front, left, right, bottom, top) --
# Mesh.cube() splits each face into 2 triangles, so each face color is
# duplicated for the pair of triangles that make it up.
back = color(255, 0, 0)
front = color(0, 255, 0)
left = color(0, 0, 255)
right = color(255, 255, 0)
bottom = color(255, 0, 255)
top = color(0, 255, 255)
face_colors = [
    back, back,
    front, front,
    left, left,
    right, right,
    bottom, bottom,
    top, top,
]

renderer.light_dir = Vector3(0.4, 0.6, 1.0)
renderer.ambient = 0.25

angle = 0.0
running = 1
while running:
    window.clear()
    model = Matrix4.rotation_y(angle).multiply(Matrix4.rotation_x(angle * 0.6))
    renderer.render_solid(mesh, model, face_colors)
    running = window.update()
    angle = angle + 0.02

window.close()
