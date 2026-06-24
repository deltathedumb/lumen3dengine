"""Shapes demo -- tests Mesh.sphere() and Mesh.cylinder() primitives.

Shows a sphere, cylinder, and cube orbiting a central point.

Controls: Escape to quit.
"""
from pugtk import Vector3, Mesh, Camera
from pugtk._renderer3d_gl import GLWindow, GLRenderer3D

from lumen3d import Instance, World, GameLoop, Material
from lumen3d._input import KEY_ESCAPE

import math

window = GLWindow("lumen3dengine shapes demo", 800, 600)
camera = Camera(
    Vector3(0.0, 5.0, 14.0),
    Vector3(0.0, 0.0, 0.0),
    Vector3(0.0, 1.0, 0.0),
    60.0,
    800.0 / 600.0,
    0.1,
    200.0,
)
renderer = GLRenderer3D(window, camera)
renderer.light_dir = Vector3(0.5, 1.0, 0.7)
renderer.ambient = 0.25

world = World(renderer)
world.collision_enabled = 0
loop = GameLoop(window, world)

sphere_mesh = Mesh.sphere(1.0, 12, 16)
cyl_mesh = Mesh.cylinder(0.6, 2.0, 12)
cube_mesh = Mesh.cube(1.0)

mat_sphere = Material("Sphere")
mat_sphere.color = 0x3388CC

mat_cyl = Material("Cylinder")
mat_cyl.color = 0xCC7733

mat_cube = Material("Cube")
mat_cube.color = 0x44AA55

mat_floor = Material("Floor")
mat_floor.color = 0x555566

sphere_inst = Instance("Sphere", sphere_mesh, [])
sphere_inst.set_material(mat_sphere)
world.add(sphere_inst)

cyl_inst = Instance("Cylinder", cyl_mesh, [])
cyl_inst.set_material(mat_cyl)
world.add(cyl_inst)

cube_inst = Instance("Cube", cube_mesh, [])
cube_inst.set_material(mat_cube)
world.add(cube_inst)

floor_mesh = Mesh.plane(20.0)
floor_inst = Instance("Floor", floor_mesh, [])
floor_inst.set_material(mat_floor)
floor_inst.position = Vector3(0.0, -2.5, 0.0)
world.add(floor_inst)


def on_update(frame: int) -> None:
    inp = loop.input
    t: float = float(frame) * 0.02
    r: float = 3.5

    sphere_inst.position = Vector3(r * math.cos(t), 0.5, r * math.sin(t))
    sphere_inst.rotation = Vector3(0.0, t, 0.0)

    cyl_inst.position = Vector3(r * math.cos(t + 2.094), 0.5, r * math.sin(t + 2.094))
    cyl_inst.rotation = Vector3(t * 0.5, t, 0.0)

    cube_inst.position = Vector3(r * math.cos(t + 4.189), 0.5, r * math.sin(t + 4.189))
    cube_inst.rotation = Vector3(t, t * 0.7, t * 0.3)

    if inp.is_key_down(KEY_ESCAPE):
        window.close()


loop.updated.connect(on_update)
loop.run()
