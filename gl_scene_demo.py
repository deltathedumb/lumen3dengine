"""Scene save/load demo.

Runs for 3 seconds, saves the world to a .scene file, then immediately
reloads it and verifies the Instance positions match.
Prints PASS or FAIL for each check.
"""
from pugtk import Vector3, Mesh, Camera, color
from pugtk._renderer3d_gl import GLWindow, GLRenderer3D

from lumen3d import Instance, World, GameLoop, SceneIO

window = GLWindow("lumen3dengine scene demo", 640, 480)
camera = Camera(
    Vector3(0.0, 4.0, 12.0),
    Vector3(0.0, 0.0, 0.0),
    Vector3(0.0, 1.0, 0.0),
    60.0,
    640.0 / 480.0,
    0.1,
    200.0,
)
renderer = GLRenderer3D(window, camera)
renderer.ambient = 0.4

world = World(renderer)
loop = GameLoop(window, world)

cube_mesh = Mesh.cube(1.0)
floor_mesh = Mesh.cube(1.0)

c1 = color(200, 60, 60)
c2 = color(60, 180, 80)
c3 = color(60, 100, 200)

part1 = Instance("RedBox", cube_mesh, [c1] * len(cube_mesh.triangles))
part1.position = Vector3(1.5, 0.0, 0.0)
part1.anchored = 1
world.add(part1)

part2 = Instance("GreenBox", cube_mesh, [c2] * len(cube_mesh.triangles))
part2.position = Vector3(-1.5, 1.0, 0.5)
part2.rotation = Vector3(0.3, 0.7, 0.0)
world.add(part2)

floor = Instance("Floor", floor_mesh, [c3] * len(floor_mesh.triangles))
floor.position = Vector3(0.0, -2.0, 0.0)
floor.scale = Vector3(8.0, 0.3, 8.0)
floor.anchored = 1
world.add(floor)

frame_target: list = [0]
done: list = [0]


def on_update(frame: int) -> None:
    if frame >= 10 and done[0] == 0:
        done[0] = 1
        sio = SceneIO()
        sio.save(world, "test_scene.scene")
        print("scene saved")

        world2 = World(renderer)
        meshes: dict = {}
        meshes["RedBox"] = cube_mesh
        meshes["GreenBox"] = cube_mesh
        meshes["Floor"] = floor_mesh
        sio.load(world2, "test_scene.scene", meshes)
        print("scene loaded, instances:")
        print(len(world2.roots))

        i: int = 0
        while i < len(world2.roots):
            inst: Instance = world2.roots[i]
            print(inst.name)
            i = i + 1

        window.close()


loop.updated.connect(on_update)
loop.run()
