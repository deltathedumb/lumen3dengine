"""Material -- named surface appearance properties for engine-layer
objects.

A Material bundles the per-surface parameters that GLRenderer3D's
Lambertian shader already supports into a single scriptable object,
so scripts can write:

    mat = Material("Brick")
    mat.color = color(180, 80, 60)
    mat.roughness = 0.9
    mat.metallic = 0.0
    part.material = mat

instead of repeating colors lists.  The renderer uses the material
to fill Instance.colors when apply_to_instance() is called, so
GLRenderer3D itself doesn't need to change.

roughness and metallic are stored on the material for future PBR-style
shaders; the current Lambertian shader only uses color and emissive.
"""
from __future__ import annotations

from pugtk._mesh import Mesh


class Material:
    """Surface appearance bundle."""
    name: str
    color: int
    emissive: int
    roughness: float
    metallic: float

    def __init__(self, name: str) -> None:
        self.name = name
        self.color = 0xC0C0C0
        self.emissive = 0x000000
        self.roughness = 0.5
        self.metallic = 0.0

    def apply_to_instance(self, mesh: Mesh, colors_out: list[int]) -> None:
        """Fill colors_out with one entry per triangle in mesh, all set to
        self.color.  colors_out is cleared first.  Call this whenever
        material.color changes, then update instance.colors = colors_out."""
        i: int = 0
        while i < len(colors_out):
            colors_out[i] = self.color
            i = i + 1
        if len(colors_out) < len(mesh.triangles):
            j: int = len(colors_out)
            while j < len(mesh.triangles):
                colors_out.append(self.color)
                j = j + 1

    def make_colors(self, mesh: Mesh) -> list[int]:
        """Return a fresh list[int] sized to mesh.triangles, filled with
        self.color -- convenience for Instance construction:
            part.colors = mat.make_colors(part.mesh)
        """
        result: list[int] = []
        i: int = 0
        while i < len(mesh.triangles):
            result.append(self.color)
            i = i + 1
        return result


class MaterialLibrary:
    """A named registry of materials -- acts as the engine-level asset
    store for surface appearances, analogous to Roblox's material enum
    but fully user-extensible.

    Usage:
        lib = MaterialLibrary()
        lib.add(Material("Brick").set_color(0xB05030))
        lib.add(Material("Metal").set_metallic(1.0).set_roughness(0.1))

        mat = lib.get("Brick")
    """
    _materials: dict

    def __init__(self) -> None:
        self._materials = {}

    def add(self, mat: Material) -> "MaterialLibrary":
        """Register a material by name. Returns self for chaining."""
        self._materials[mat.name] = mat
        return self

    def get(self, name: str) -> Material:
        """Look up a material by name.  Returns None if not found."""
        return self._materials.get(name)

    def names(self) -> list[str]:
        return self._materials.keys()
