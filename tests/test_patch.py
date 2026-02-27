"""Tests for manim monkey-patching."""

from __future__ import annotations


def test_renderer_type_has_metal():
    """RendererType enum should have a METAL member after patching."""
    from manim.constants import RendererType

    assert hasattr(RendererType, "METAL")
    assert RendererType.METAL.value == "metal"
    assert RendererType("metal") is RendererType.METAL


def test_config_renderer_accepts_metal():
    """config.renderer = 'metal' should not raise."""
    from manim import config
    from manim.constants import RendererType

    original = config.renderer
    try:
        config.renderer = "metal"
        assert config.renderer == RendererType.METAL
        assert config.renderer.value == "metal"
    finally:
        config.renderer = original


def test_config_renderer_case_insensitive():
    """config.renderer = 'Metal' should work (case-insensitive)."""
    from manim import config
    from manim.constants import RendererType

    original = config.renderer
    try:
        config.renderer = "Metal"
        assert config.renderer == RendererType.METAL
    finally:
        config.renderer = original


def test_existing_renderers_still_work():
    """Cairo and OpenGL enum values should still work."""
    from manim.constants import RendererType

    assert RendererType("cairo") is RendererType.CAIRO
    assert RendererType("opengl") is RendererType.OPENGL
