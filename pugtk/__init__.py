from ._vector import Vector2, Vector3
from ._shapes import PointShape, ObjectDefinitions
from ._window import Window, color, shade, mix_colors
from ._matrix import Matrix4
from ._camera import Camera
from ._mesh import Mesh
from ._texture import Texture
from ._shading import (
    ShaderContext,
    unlit_shader,
    lambert_shader,
    phong_shader,
    checker_shader,
    textured_shader,
    pbr_shader,
    fresnel_rim_shader,
)
from ._scene import SceneObject
from ._renderer3d import Renderer3D
from ._light import PointLight
from ._mesh_loader import load_obj
from ._gltf_loader import load_gltf
