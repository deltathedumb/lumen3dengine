"""Physics + collision demo for lumen3dengine.

A red player cube falls under gravity and can be controlled with WASD.
A grey static floor and four walls form a box arena. When the player
lands the touched signal fires. Collision response is handled entirely
by the engine (World.step) -- no manual floor clamp needed in script.

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

MOVE_SPEED: float = 5.0
JUMP_VEL: float = 6.0

window = GLWindow("lumen3dengine physics demo", 800, 600)
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
renderer.light_dir = Vector3(0.5, 0.8, 0.6)
renderer.ambient = 0.3

world = World(renderer)
loop = GameLoop(window, world)

player_mesh = Mesh.cube(1.0)
player_color = color(220, 60, 60)
player = Instance("Player", player_mesh, [player_color] * len(player_mesh.triangles))
player.gravity_enabled = 1
player.position = Vector3(0.0, 5.0, 0.0)
world.add(player)

floor_mesh = Mesh.cube(1.0)
floor_color = color(80, 80, 90)
floor_inst = Instance("Floor", floor_mesh, [floor_color] * len(floor_mesh.triangles))
floor_inst.anchored = 1
floor_inst.scale = Vector3(14.0, 0.5, 14.0)
floor_inst.position = Vector3(0.0, -3.25, 0.0)
world.add(floor_inst)

wall_mesh = Mesh.cube(1.0)
wall_color = color(60, 100, 150)

wall_n = Instance("WallN", wall_mesh, [wall_color] * len(wall_mesh.triangles))
wall_n.anchored = 1
wall_n.scale = Vector3(14.0, 6.0, 0.5)
wall_n.position = Vector3(0.0, 0.0, -7.0)
world.add(wall_n)

wall_s = Instance("WallS", wall_mesh, [wall_color] * len(wall_mesh.triangles))
wall_s.anchored = 1
wall_s.scale = Vector3(14.0, 6.0, 0.5)
wall_s.position = Vector3(0.0, 0.0, 7.0)
world.add(wall_s)

wall_e = Instance("WallE", wall_mesh, [wall_color] * len(wall_mesh.triangles))
wall_e.anchored = 1
wall_e.scale = Vector3(0.5, 6.0, 14.0)
wall_e.position = Vector3(7.0, 0.0, 0.0)
world.add(wall_e)

wall_w = Instance("WallW", wall_mesh, [wall_color] * len(wall_mesh.triangles))
wall_w.anchored = 1
wall_w.scale = Vector3(0.5, 6.0, 14.0)
wall_w.position = Vector3(-7.0, 0.0, 0.0)
world.add(wall_w)

box_mesh = Mesh.cube(1.0)
box_color = color(200, 160, 60)
box1 = Instance("Box1", box_mesh, [box_color] * len(box_mesh.triangles))
box1.gravity_enabled = 1
box1.position = Vector3(2.0, 2.0, 0.0)
world.add(box1)

box2 = Instance("Box2", box_mesh, [box_color] * len(box_mesh.triangles))
box2.gravity_enabled = 1
box2.position = Vector3(-2.0, 4.0, 1.0)
world.add(box2)

on_floor: list = [0]
landing_count: list = [0]
was_touching: list = [0]


def on_player_touched(other_idx: int) -> None:
    if was_touching[0] == 0:
        landing_count[0] = landing_count[0] + 1
        print("collision, count:")
        print(landing_count[0])
        was_touching[0] = 1


player.touched.connect(on_player_touched)


def on_update(frame: int) -> None:
    inp = loop.input

    vx: float = player.velocity.x
    vz: float = player.velocity.z

    if inp.is_key_down(KEY_A):
        vx = -MOVE_SPEED
    elif inp.is_key_down(KEY_D):
        vx = MOVE_SPEED
    else:
        vx = vx * 0.75

    if inp.is_key_down(KEY_W):
        vz = -MOVE_SPEED
    elif inp.is_key_down(KEY_S):
        vz = MOVE_SPEED
    else:
        vz = vz * 0.75

    grounded: int = on_floor[0]
    if inp.is_key_down(KEY_SPACE) and grounded == 1:
        player.velocity = Vector3(vx, JUMP_VEL, vz)
    else:
        player.velocity = Vector3(vx, player.velocity.y, vz)

    py: float = player.position.y
    if py < -2.5:
        on_floor[0] = 1
    else:
        on_floor[0] = 0
        was_touching[0] = 0

    if inp.is_key_down(KEY_ESCAPE):
        window.close()


loop.updated.connect(on_update)
loop.run()
