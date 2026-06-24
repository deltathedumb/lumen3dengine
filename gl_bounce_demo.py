"""Bounce demo -- tests Instance.restitution (elastic collision).

Three spheres with different bounciness drop onto an anchored floor:
  Left (blue):   restitution=0.0 -- thuds, no bounce
  Center (red):  restitution=0.5 -- medium bounce
  Right (green): restitution=0.9 -- very bouncy

Controls: Escape to quit, R to reset positions.
"""
from pugtk import Vector3, Mesh, Camera
from pugtk._renderer3d_gl import GLWindow, GLRenderer3D

from lumen3d import Instance, World, GameLoop, Material
from lumen3d._input import KEY_ESCAPE, KEY_R

window = GLWindow("lumen3dengine bounce demo", 900, 600)
camera = Camera(
    Vector3(0.0, 4.0, 16.0),
    Vector3(0.0, 0.0, 0.0),
    Vector3(0.0, 1.0, 0.0),
    60.0,
    900.0 / 600.0,
    0.1,
    200.0,
)
renderer = GLRenderer3D(window, camera)
renderer.light_dir = Vector3(0.4, 1.0, 0.6)
renderer.ambient = 0.2

world = World(renderer)
loop = GameLoop(window, world)

sphere_mesh = Mesh.sphere(0.7, 10, 14)
floor_mesh = Mesh.plane(20.0)

mat_floor = Material("Floor")
mat_floor.color = 0x445544

mat_dead = Material("Dead")
mat_dead.color = 0x3366AA

mat_med = Material("Med")
mat_med.color = 0xCC4433

mat_bouncy = Material("Bouncy")
mat_bouncy.color = 0x33AA44

floor = Instance("Floor", floor_mesh, [])
floor.set_material(mat_floor)
floor.anchored = 1
floor.restitution = 0.0
world.add(floor)

ball_dead = Instance("BallDead", sphere_mesh, [])
ball_dead.set_material(mat_dead)
ball_dead.gravity_enabled = 1
ball_dead.restitution = 0.0
ball_dead.position = Vector3(-4.0, 6.0, 0.0)
world.add(ball_dead)

ball_med = Instance("BallMed", sphere_mesh, [])
ball_med.set_material(mat_med)
ball_med.gravity_enabled = 1
ball_med.restitution = 0.5
ball_med.position = Vector3(0.0, 6.0, 0.0)
world.add(ball_med)

ball_bouncy = Instance("BallBouncy", sphere_mesh, [])
ball_bouncy.set_material(mat_bouncy)
ball_bouncy.gravity_enabled = 1
ball_bouncy.restitution = 0.9
ball_bouncy.position = Vector3(4.0, 6.0, 0.0)
world.add(ball_bouncy)


def reset() -> None:
    ball_dead.position = Vector3(-4.0, 6.0, 0.0)
    ball_dead.velocity = Vector3(0.0, 0.0, 0.0)
    ball_med.position = Vector3(0.0, 6.0, 0.0)
    ball_med.velocity = Vector3(0.0, 0.0, 0.0)
    ball_bouncy.position = Vector3(4.0, 6.0, 0.0)
    ball_bouncy.velocity = Vector3(0.0, 0.0, 0.0)


def on_update(frame: int) -> None:
    inp = loop.input
    if inp.is_key_down(KEY_R) == 1:
        reset()
    if inp.is_key_down(KEY_ESCAPE) == 1:
        window.close()


loop.updated.connect(on_update)
loop.run()
