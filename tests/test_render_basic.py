"""Basic rendering tests using the Metal renderer."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def metal_config():
    """Set config to use Metal renderer, restore after test."""
    from manim import config

    original = config.renderer
    config.renderer = "metal"
    yield config
    config.renderer = original


def test_scene_init_with_metal_renderer(metal_config):
    """Scene() should instantiate a MetalRenderer when config.renderer is metal."""
    from manim import Scene

    from manim_metal.renderer import MetalRenderer

    scene = Scene()
    assert isinstance(scene.renderer, MetalRenderer)
    assert scene.renderer.camera is not None
    assert scene.renderer.num_plays == 0


def test_metal_camera_produces_valid_frame(metal_config):
    """MetalCamera.get_frame() should return a valid RGBA numpy array."""
    from manim_metal.metal_camera import MetalCamera

    cam = MetalCamera()
    cam.reset()
    frame = np.array(cam.pixel_array)

    assert frame.dtype == np.uint8
    assert frame.shape == (cam.pixel_height, cam.pixel_width, 4)
    # Background should not be all zeros (default manim bg is dark)
    assert frame.shape[0] > 0 and frame.shape[1] > 0


def test_render_circle(metal_config):
    """Render a Circle and verify the frame has non-background pixels."""
    from manim import Circle, Scene

    from manim_metal.renderer import MetalRenderer

    scene = Scene()
    assert isinstance(scene.renderer, MetalRenderer)

    circle = Circle()
    scene.add(circle)
    scene.renderer.update_frame(scene)
    frame = scene.renderer.get_frame()

    assert frame.shape == (
        scene.renderer.camera.pixel_height,
        scene.renderer.camera.pixel_width,
        4,
    )
    # The frame should have some non-background pixels (the circle)
    bg = scene.renderer.camera.background
    diff = np.any(frame != bg, axis=2)
    assert diff.any(), "Frame should contain rendered circle pixels different from background"
