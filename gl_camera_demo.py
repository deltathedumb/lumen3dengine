"""Third-person camera demo.

A sphere (the "player") sits on a floor.  The camera orbits the player:
  - Right-drag mouse to rotate the orbit camera
  - WASD to move the player (relative to camera yaw)
  - Space to jump
  - Escape to quit

This demo also tests ThirdPersonCamera.forward_dir()/right_dir() via
cam.yaw to get camera-relative movement directions.
"""
from pugtk import Vector3, Mesh, Camera
from pugtk._renderer3d_gl import GLWindow, GLRenderer3D

from lumen3d import (
    Instance, World, GameLoop, Material,
    ThirdPersonCamera,
)
from lumen3d._input import (
    KEY_ESCAPE, KEY_W, KEY_A, KEY_S, KEY_D, KEY_SPACE,
)

import math

SPEED: float = 5.0
JUMP_VEL: float = 6.0

window = GLWindow("lumen3dengine camera demo", 900, 600)
camera = Camera(
    Vector3(0.0, 5.0, 10.0),
    Vector3(0.0, 0.0, 0.0),
    Vector3(0.0, 1.0, 0.0),
    60.0,
    900.0 / 600.0,
    0.1,
    200.0,
)
renderer = GLRenderer3D(window, camera)
renderer.light_dir = Vector3(0.6, 1.0, 0.5)
renderer.ambient = 0.25

world = World(renderer)
loop = GameLoop(window, world)

player_mesh = Mesh.sphere(0.6, 10, 14)
mat_player = Material("Player")
mat_player.color = 0x44AADD

player = Instance("Player", player_mesh, [])
player.set_material(mat_player)
player.gravity_enabled = 1
player.position = Vector3(0.0, 1.0, 0.0)
world.add(player)

floor_mesh = Mesh.plane(30.0)
mat_floor = Material("Floor")
mat_floor.color = 0x667766
floor = Instance("Floor", floor_mesh, [])
floor.set_material(mat_floor)
floor.anchored = 1
world.add(floor)

box1_mesh = Mesh.cube(1.5)
mat_box = Material("Box")
mat_box.color = 0xCC8844
box1 = Instance("Box1", box1_mesh, [])
box1.set_material(mat_box)
box1.anchored = 1
box1.position = Vector3(4.0, 0.75, 2.0)
world.add(box1)

box2 = Instance("Box2", box1_mesh, [])
box2.set_material(mat_box)
box2.anchored = 1
box2.position = Vector3(-3.5, 0.75, -2.5)
world.add(box2)

cyl_mesh = Mesh.cylinder(0.5, 3.0, 12)
mat_cyl = Material("Pillar")
mat_cyl.color = 0xAA9988
cyl = Instance("Pillar", cyl_mesh, [])
cyl.set_material(mat_cyl)
cyl.anchored = 1
cyl.position = Vector3(0.0, 1.5, -5.0)
world.add(cyl)

cam_ctrl = ThirdPersonCamera(camera, player)
cam_ctrl.distance = 7.0
cam_ctrl.height = 1.2


def on_update(frame: int) -> None:
    inp = loop.input
    cam_ctrl.update(inp)

    yaw: float = cam_ctrl.yaw
    fwd_x: float = math.sin(yaw)
    fwd_z: float = math.cos(yaw)
    right_x: float = math.cos(yaw)
    right_z: float = -math.sin(yaw)

    move_x: float = 0.0
    move_z: float = 0.0
    if inp.is_key_down(KEY_W) == 1:
        move_x = move_x + fwd_x
        move_z = move_z + fwd_z
    if inp.is_key_down(KEY_S) == 1:
        move_x = move_x - fwd_x
        move_z = move_z - fwd_z
    if inp.is_key_down(KEY_A) == 1:
        move_x = move_x - right_x
        move_z = move_z - right_z
    if inp.is_key_down(KEY_D) == 1:
        move_x = move_x + right_x
        move_z = move_z + right_z

    dt: float = loop.fixed_dt
    player.velocity = Vector3(
        move_x * SPEED,
        player.velocity.y,
        move_z * SPEED,
    )

    if inp.is_key_down(KEY_SPACE) == 1:
        if player.velocity.y > -0.5 and player.velocity.y < 0.5:
            player.velocity = Vector3(player.velocity.x, JUMP_VEL, player.velocity.z)

    if inp.is_key_down(KEY_ESCAPE) == 1:
        window.close()


loop.updated.connect(on_update)
loop.run()
