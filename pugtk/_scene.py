from ._matrix import Matrix4
from ._mesh import Mesh
from ._texture import Texture


class SceneObject:
    """One mesh instance in a multi-object scene: its own transform, base
    face colors, and texture. Renderer3D.render_scene() depth-sorts faces
    *across* every SceneObject in the list together (not each object's
    faces sorted only against themselves, then drawn object-by-object),
    so a nearer object's face correctly covers a farther object's face
    behind it -- painter's algorithm extended to multiple meshes instead
    of just one mesh's own faces.

    This still has painter's algorithm's usual limitation: it orders
    whole faces by centroid distance, so two faces that actually
    interpenetrate in 3D (rather than just one being farther away) can
    still sort wrong. Fine for separate, non-overlapping objects.
    """

    mesh: Mesh
    model: Matrix4
    colors: list[int]
    texture: Texture

    def __init__(self, mesh: Mesh, model: Matrix4, colors: list[int], texture: Texture):
        self.mesh = mesh
        self.model = model
        self.colors = colors
        self.texture = texture
