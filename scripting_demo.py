from pugtk import Vector3, Mesh, Camera, color
from pugtk._renderer3d_gl import GLWindow, GLRenderer3D

from lumen3d import Instance, World, GameLoop

window = GLWindow("lumen3dengine scripting demo", 640, 480)
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

world = World(renderer)
loop = GameLoop(window, world)

cube_mesh = Mesh.cube(2.0)
cube_color = color(80, 200, 120)
part = Instance("Cube", cube_mesh, [cube_color] * len(cube_mesh.triangles))
world.add(part)


def on_lap(turns: int) -> None:
    print("lap complete, total laps:")
    print(turns)


part.touched.connect(on_lap)

laps: list = [0]


def on_update(frame: int) -> None:
    angle: float = float(frame) * 0.02
    part.rotation = Vector3(angle * 0.6, angle, 0.0)
    full_turns: int = int(angle / 6.28318)
    if full_turns > laps[0]:
        laps[0] = full_turns
        part.touched(full_turns)


loop.updated.connect(on_update)
loop.run()
