"""Physics + collision demo for lumen3dengine.

A red player cube falls under gravity and can be controlled with WASD.
A grey static floor sits below it. When the player lands on the floor
the `touched` signal fires and prints a message. The player stops
sinking through the floor via a simple position-clamp collision response.

Controls:
  W/S     -- move forward/back (Z axis)
  A/D     -- move left/right (X axis)
  Space   -- jump (sets upward velocity when grounded)
  Escape  -- quit
"""
from pugtk import Vector3, Mesh, Camera, color
from pugtk._renderer3d_gl import GLWindow, GLRenderer3D

from lumen3d import Instance, World, GameLoop
from lumen3d._input import KEY_W, KEY_S, KEY_A, KEY_D, KEY_SPACE, KEY_ESCAPE

MOVE_SPEED: float = 4.0
JUMP_VEL: float = 5.0
FLOOR_Y: float = -2.5

window = GLWindow("lumen3dengine physics demo", 800, 600)
camera = Camera(
    Vector3(0.0, 4.0, 12.0),
    Vector3(0.0, 0.0, 0.0),
    Vector3(0.0, 1.0, 0.0),
    60.0,
    800.0 / 600.0,
    0.1,
    200.0,
)
renderer = GLRenderer3D(window, camera)
renderer.light_dir = Vector3(0.5, 0.8, 0.6)
renderer.ambient = 0.3

world = World(renderer)
loop = GameLoop(window, world)

player_mesh = Mesh.cube(1.0)
player_color = color(220, 60, 60)
player = Instance("Player", player_mesh, [player_color] * len(player_mesh.triangles))
player.gravity_enabled = 1
player.position = Vector3(0.0, 4.0, 0.0)
world.add(player)

floor_mesh = Mesh.cube(1.0)
floor_color = color(120, 120, 130)
floor_inst = Instance("Floor", floor_mesh, [floor_color] * len(floor_mesh.triangles))
floor_inst.anchored = 1
floor_inst.scale = Vector3(12.0, 0.4, 12.0)
floor_inst.position = Vector3(0.0, -3.0, 0.0)
world.add(floor_inst)

on_floor: list = [0]
bounce_count: list = [0]
was_touching: list = [0]


def on_player_touched(other_idx: int) -> None:
    if was_touching[0] == 0:
        bounce_count[0] = bounce_count[0] + 1
        print("player landed, landing count:")
        print(bounce_count[0])
        was_touching[0] = 1


player.touched.connect(on_player_touched)


def on_update(frame: int) -> None:
    inp = loop.input
    dt: float = loop.fixed_dt

    vx: float = player.velocity.x
    vz: float = player.velocity.z

    if inp.is_key_down(KEY_A):
        vx = -MOVE_SPEED
    elif inp.is_key_down(KEY_D):
        vx = MOVE_SPEED
    else:
        vx = vx * 0.8

    if inp.is_key_down(KEY_W):
        vz = -MOVE_SPEED
    elif inp.is_key_down(KEY_S):
        vz = MOVE_SPEED
    else:
        vz = vz * 0.8

    if inp.is_key_down(KEY_SPACE) and on_floor[0] == 1:
        player.velocity = Vector3(vx, JUMP_VEL, vz)
    else:
        player.velocity = Vector3(vx, player.velocity.y, vz)

    py: float = player.position.y
    if py <= FLOOR_Y:
        player.position = Vector3(player.position.x, FLOOR_Y, player.position.z)
        player.velocity = Vector3(player.velocity.x, 0.0, player.velocity.z)
        on_floor[0] = 1
    else:
        on_floor[0] = 0
        was_touching[0] = 0

    if inp.is_key_down(KEY_ESCAPE):
        window.close()


loop.updated.connect(on_update)
loop.run()
