from pugtk import Window, Vector3, Matrix4, Camera, Mesh, Renderer3D, color, textured_shader, Texture

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
renderer.pixel_shader = textured_shader
renderer.light_dir = Vector3(0.4, 0.6, 1.0)
renderer.ambient = 0.25
mesh = Mesh.cube(2.0)

# A small procedural test texture -- a colored grid with a diagonal
# stripe, picked because both axes and the diagonal make it obvious if UV
# mapping is flipped/transposed/sheared. Real PNG loading is left to
# whoever's loading game assets; Texture just needs the decoded pixels.
tex_size = 64
tex_pixels: list[int] = []
ty = 0
while ty < tex_size:
    tx = 0
    while tx < tex_size:
        c = color(40, 40, 80)
        if (tx // 8 + ty // 8) % 2 == 0:
            c = color(220, 220, 240)
        if abs(tx - ty) < 3:
            c = color(255, 60, 60)
        if tx < 4 or ty < 4:
            c = color(60, 220, 60)
        tex_pixels.append(c)
        tx = tx + 1
    ty = ty + 1
texture = Texture(tex_size, tex_size, tex_pixels)

# Mesh.cube() splits each face into 2 triangles, so each face color is
# duplicated for the pair of triangles that make it up.
face_colors = [
    color(255, 0, 0), color(255, 0, 0),
    color(0, 255, 0), color(0, 255, 0),
    color(0, 0, 255), color(0, 0, 255),
    color(255, 255, 0), color(255, 255, 0),
    color(255, 0, 255), color(255, 0, 255),
    color(0, 255, 255), color(0, 255, 255),
]

angle = 0.0
running = 1
while running:
    window.clear()
    model = Matrix4.rotation_y(angle).multiply(Matrix4.rotation_x(angle * 0.6))
    renderer.render_solid_per_pixel(mesh, model, face_colors, texture)
    running = window.update()
    angle = angle + 0.02

window.close()
