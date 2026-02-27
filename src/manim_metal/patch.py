"""Monkey-patches manim to support the Metal renderer.

Patches three things at import time:
1. Adds METAL member to RendererType enum
2. The ManimConfig.renderer setter already works once the enum is extended
3. Wraps Scene.__init__ to instantiate MetalRenderer when config.renderer == "metal"
"""

from __future__ import annotations

import functools

_patched = False


def _extend_renderer_type_enum() -> None:
    """Add METAL = "metal" to manim.constants.RendererType."""
    from manim.constants import RendererType

    if hasattr(RendererType, "METAL"):
        return

    # Create the new enum member manually
    new_member = object.__new__(RendererType)
    new_member._name_ = "METAL"
    new_member._value_ = "metal"

    # Register in the enum's internal mappings
    RendererType._member_map_["METAL"] = new_member
    RendererType._value2member_map_["metal"] = new_member
    RendererType._member_names_.append("METAL")

    # Python 3.12+ stores enum members directly in __dict__.
    # EnumMeta.__setattr__ blocks reassignment, so bypass it via type.
    type.__setattr__(RendererType, "METAL", new_member)


def _patch_scene_init() -> None:
    """Wrap Scene.__init__ to instantiate MetalRenderer for metal renderer type."""
    from manim.scene.scene import Scene

    original_init = Scene.__init__

    @functools.wraps(original_init)
    def patched_init(self, renderer=None, **kwargs):
        from manim import config
        from manim.constants import RendererType

        if renderer is None and config.renderer == RendererType.METAL:
            from manim_metal.renderer import MetalRenderer

            renderer = MetalRenderer(skip_animations=kwargs.get("skip_animations", False))
        return original_init(self, renderer=renderer, **kwargs)

    Scene.__init__ = patched_init


def _patch_scene_cairo_assertions() -> None:
    """Patch Scene methods that assert config.renderer == CAIRO.

    Metal uses Cairo-style Mobjects (not OpenGL), so it takes the same
    code path as Cairo. The assertions ``assert config.renderer == RendererType.CAIRO``
    in ``add``, ``remove``, and ``get_mobject_family_members`` must accept METAL too.
    """
    from manim import config
    from manim.constants import RendererType
    from manim.scene.scene import Scene

    for method_name in ("add", "remove", "get_mobject_family_members"):
        original = getattr(Scene, method_name)

        @functools.wraps(original)
        def patched(self, *args, _orig=original, **kwargs):
            # Temporarily pretend we're Cairo so the assertions pass
            should_swap = config.renderer == RendererType.METAL
            if should_swap:
                config._d["renderer"] = RendererType.CAIRO
            try:
                return _orig(self, *args, **kwargs)
            finally:
                if should_swap:
                    config._d["renderer"] = RendererType.METAL

        setattr(Scene, method_name, patched)


def apply_patches() -> None:
    """Apply all monkey-patches. Safe to call multiple times."""
    global _patched
    if _patched:
        return
    _extend_renderer_type_enum()
    _patch_scene_init()
    _patch_scene_cairo_assertions()
    _patched = True
