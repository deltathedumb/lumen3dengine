"""Tag system -- label Instances with string tags for group queries.

Equivalent to Roblox's CollectionService / Unity's GameObject.tag.
Tags are stored on a global registry (dict[str, list[Instance]]),
not on Instance itself, to avoid modifying the Instance class.

Usage:
    tag(player, "Player")
    tag(enemy1, "Enemy")
    tag(enemy2, "Enemy")
    tag(floor, "Ground")

    enemies = get_tagged("Enemy")     # -> list of all Enemy instances

    has_tag(player, "Player")         # -> 1
    remove_tag(player, "Player")
    clear_tags(enemy1)                # remove all tags from enemy1
"""
from __future__ import annotations

from ._instance import Instance


_tag_registry: dict = {}


def tag(inst: Instance, tag_name: str) -> None:
    """Add tag_name to inst. Idempotent (safe to call multiple times)."""
    existing = _tag_registry.get(tag_name)
    if existing is None:
        _tag_registry[tag_name] = [inst]
        return
    i: int = 0
    while i < len(existing):
        if existing[i].name == inst.name and existing[i] is inst:
            return
        i = i + 1
    existing.append(inst)


def untag(inst: Instance, tag_name: str) -> None:
    """Remove tag_name from inst (no-op if not tagged)."""
    existing = _tag_registry.get(tag_name)
    if existing is None:
        return
    new_list: list = []
    i: int = 0
    while i < len(existing):
        if existing[i] is not inst:
            new_list.append(existing[i])
        i = i + 1
    _tag_registry[tag_name] = new_list


def has_tag(inst: Instance, tag_name: str) -> int:
    """Return 1 if inst has the given tag, 0 otherwise."""
    existing = _tag_registry.get(tag_name)
    if existing is None:
        return 0
    i: int = 0
    while i < len(existing):
        if existing[i] is inst:
            return 1
        i = i + 1
    return 0


def get_tagged(tag_name: str) -> list:
    """Return a list of all Instances with the given tag (may be empty)."""
    existing = _tag_registry.get(tag_name)
    if existing is None:
        return []
    result: list = []
    i: int = 0
    while i < len(existing):
        result.append(existing[i])
        i = i + 1
    return result


def clear_tags(inst: Instance) -> None:
    """Remove all tags from inst across the entire registry."""
    keys: list = list(_tag_registry.keys())
    i: int = 0
    while i < len(keys):
        untag(inst, keys[i])
        i = i + 1


def all_tags(inst: Instance) -> list:
    """Return all tag names currently applied to inst."""
    result: list = []
    keys: list = list(_tag_registry.keys())
    i: int = 0
    while i < len(keys):
        if has_tag(inst, keys[i]) == 1:
            result.append(keys[i])
        i = i + 1
    return result


def tag_count(tag_name: str) -> int:
    """Return how many Instances have the given tag."""
    existing = _tag_registry.get(tag_name)
    if existing is None:
        return 0
    return len(existing)


def clear_registry() -> None:
    """Remove all tags from all instances. Useful when loading a new scene."""
    keys: list = list(_tag_registry.keys())
    i: int = 0
    while i < len(keys):
        _tag_registry[keys[i]] = []
        i = i + 1
