"""CameraController -- game-camera controllers for lumen3dengine.

Wraps pugtk's Camera to add game-logic camera behaviors:

  ThirdPersonCamera   -- orbits a target Instance, mouse look, zoom
  FirstPersonCamera   -- moves with an Instance, mouse look

Both are updated each frame by calling update() in an updated callback.

Example:
    cam_ctrl = ThirdPersonCamera(camera, player)
    cam_ctrl.distance = 8.0
    cam_ctrl.height = 3.0

    def on_update(frame: int) -> None:
        cam_ctrl.update(loop.input)

    loop.updated.connect(on_update)
"""
from __future__ import annotations

import math

from pugtk._camera import Camera
from pugtk._vector import Vector3

from ._instance import Instance
from ._input import Input


class ThirdPersonCamera:
    """Orbits a target Instance at a fixed distance/height, with optional
    mouse-drag to rotate around the target.

    yaw: horizontal orbit angle in radians (0 = behind target)
    pitch: vertical angle in radians (clamped to avoid gimbal at poles)
    distance: how far the camera sits from the target center
    height: extra Y offset applied to the look-at target point
    """

    camera: Camera
    target: Instance
    yaw: float
    pitch: float
    distance: float
    height: float
    mouse_sensitivity: float
    _prev_mouse_x: int
    _prev_mouse_y: int
    _mouse_dragging: int

    def __init__(self, camera: Camera, target: Instance) -> None:
        self.camera = camera
        self.target = target
        self.yaw = 0.0
        self.pitch = 0.3
        self.distance = 8.0
        self.height = 1.5
        self.mouse_sensitivity = 0.005
        self._prev_mouse_x = 0
        self._prev_mouse_y = 0
        self._mouse_dragging = 0

    def update(self, inp: Input) -> None:
        """Call once per frame from your updated callback.  Reads mouse
        right-button drag to orbit; updates camera.position and
        camera.target each frame from current yaw/pitch/distance."""
        if inp.is_mouse_down(3) == 1:
            if self._mouse_dragging == 0:
                self._prev_mouse_x = inp.mouse_x
                self._prev_mouse_y = inp.mouse_y
                self._mouse_dragging = 1
            else:
                dx: int = inp.mouse_x - self._prev_mouse_x
                dy: int = inp.mouse_y - self._prev_mouse_y
                self.yaw = self.yaw + float(dx) * self.mouse_sensitivity
                self.pitch = self.pitch - float(dy) * self.mouse_sensitivity
                if self.pitch < 0.05:
                    self.pitch = 0.05
                if self.pitch > 1.5:
                    self.pitch = 1.5
                self._prev_mouse_x = inp.mouse_x
                self._prev_mouse_y = inp.mouse_y
        else:
            self._mouse_dragging = 0

        tx: float = self.target.position.x
        ty: float = self.target.position.y
        tz: float = self.target.position.z

        cam_x: float = tx + self.distance * math.cos(self.pitch) * math.sin(self.yaw)
        cam_y: float = ty + self.distance * math.sin(self.pitch) + self.height
        cam_z: float = tz + self.distance * math.cos(self.pitch) * math.cos(self.yaw)

        self.camera.position = Vector3(cam_x, cam_y, cam_z)
        self.camera.target = Vector3(tx, ty + self.height, tz)

    def set_behind(self, instance: Instance) -> None:
        """Snap yaw so the camera sits directly behind instance, based
        on instance.rotation.y."""
        self.yaw = instance.rotation.y + math.pi


class FirstPersonCamera:
    """Attaches to an Instance's position and rotates with mouse movement.

    The camera eye is placed at (position + eye_offset) and looks in the
    direction controlled by yaw (mouse X) / pitch (mouse Y). Ideal for
    an FPS character or a vehicle cabin view.
    """

    camera: Camera
    host: Instance
    yaw: float
    pitch: float
    eye_offset: Vector3
    mouse_sensitivity: float
    _prev_mouse_x: int
    _prev_mouse_y: int

    def __init__(self, camera: Camera, host: Instance) -> None:
        self.camera = camera
        self.host = host
        self.yaw = 0.0
        self.pitch = 0.0
        self.eye_offset = Vector3(0.0, 0.8, 0.0)
        self.mouse_sensitivity = 0.003
        self._prev_mouse_x = 0
        self._prev_mouse_y = 0

    def update(self, inp: Input) -> None:
        dx: int = inp.mouse_x - self._prev_mouse_x
        dy: int = inp.mouse_y - self._prev_mouse_y
        self.yaw = self.yaw + float(dx) * self.mouse_sensitivity
        self.pitch = self.pitch - float(dy) * self.mouse_sensitivity
        if self.pitch < -1.4:
            self.pitch = -1.4
        if self.pitch > 1.4:
            self.pitch = 1.4
        self._prev_mouse_x = inp.mouse_x
        self._prev_mouse_y = inp.mouse_y

        hx: float = self.host.position.x
        hy: float = self.host.position.y
        hz: float = self.host.position.z

        eye_x: float = hx + self.eye_offset.x
        eye_y: float = hy + self.eye_offset.y
        eye_z: float = hz + self.eye_offset.z

        look_x: float = eye_x + math.cos(self.pitch) * math.sin(self.yaw)
        look_y: float = eye_y + math.sin(self.pitch)
        look_z: float = eye_z + math.cos(self.pitch) * math.cos(self.yaw)

        self.camera.position = Vector3(eye_x, eye_y, eye_z)
        self.camera.target = Vector3(look_x, look_y, look_z)

    def forward_dir(self) -> Vector3:
        """Returns the horizontal forward direction vector (ignores pitch)
        so movement code can use it: velocity = forward_dir() * speed."""
        x: float = math.sin(self.yaw)
        z: float = math.cos(self.yaw)
        return Vector3(x, 0.0, z)

    def right_dir(self) -> Vector3:
        """Returns the horizontal right strafe direction."""
        x: float = math.cos(self.yaw)
        z: float = -math.sin(self.yaw)
        return Vector3(x, 0.0, z)
