"""Basic 3D rendering tests for the Metal renderer."""

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


def test_3d_camera_init(metal_config):
    """MetalCamera should have ThreeDCamera-compatible ValueTracker properties."""
    from manim_metal.metal_camera import MetalCamera

    cam = MetalCamera()

    # ValueTracker properties should exist and have default values
    assert cam.phi_tracker.get_value() == 0
    assert cam.theta_tracker.get_value() == 0
    assert cam.gamma_tracker.get_value() == 0
    assert cam.zoom_tracker.get_value() == 1
    assert cam.focal_distance_tracker.get_value() == 20.0

    # Setters should work
    cam.set_phi(1.0)
    assert cam.phi_tracker.get_value() == 1.0
    cam.set_theta(-0.5)
    assert cam.theta_tracker.get_value() == -0.5

    # get_value_trackers should return 5 trackers
    trackers = cam.get_value_trackers()
    assert len(trackers) == 5


def test_3d_camera_is_3d_detection(metal_config):
    """Camera should detect 3D mode when phi/theta/gamma are non-zero."""
    from manim_metal.metal_camera import MetalCamera

    cam = MetalCamera()
    assert not cam._is_3d_active

    cam.set_phi(0.5)
    assert cam._is_3d_active


def test_rotation_matrix(metal_config):
    """build_rotation_matrix should produce a valid 3x3 rotation matrix."""
    from manim_metal.utils import build_rotation_matrix

    # Identity case: phi=0, theta=-PI/2, gamma=0 (default camera)
    R = build_rotation_matrix(0, -np.pi / 2, 0)
    assert R.shape == (3, 3)
    # Should be orthogonal: R @ R.T ≈ I
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    # Should have determinant 1
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_uniform_packing_2d(metal_config):
    """2D uniform packing should produce the correct buffer layout."""
    from manim_metal.metal_camera import MetalCamera

    cam = MetalCamera()
    color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    buf = cam._make_uniform_data(color)

    # Check MVP is at offset 0
    mvp = buf[:16].reshape(4, 4)
    assert mvp[0, 0] != 0  # should have scale factors

    # Check color at offset 16
    assert np.allclose(buf[16:20], color)

    # is_3d flag should be 0
    assert buf.view(np.uint32)[40] == 0


def test_uniform_packing_3d(metal_config):
    """3D uniform packing should include rotation, frame_center, etc."""
    from manim_metal.metal_camera import MetalCamera

    cam = MetalCamera()
    cam.set_phi(0.5)
    cam.set_theta(-0.3)
    cam.reset_rotation_matrix()

    color = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    buf = cam._make_uniform_data(color)

    # is_3d flag should be 1
    assert buf.view(np.uint32)[40] == 1

    # Rotation should be packed at float indices 20-31
    # Column 0 at 20-22, Column 1 at 24-26, Column 2 at 28-30
    rot = cam._rotation_matrix
    assert np.allclose(buf[20:23], rot[:, 0].astype(np.float32), atol=1e-6)
    assert np.allclose(buf[24:27], rot[:, 1].astype(np.float32), atol=1e-6)
    assert np.allclose(buf[28:31], rot[:, 2].astype(np.float32), atol=1e-6)


def test_3d_scene_patch(metal_config):
    """ThreeDScene should be importable and its methods should accept METAL renderer."""
    from manim.scene.three_d_scene import ThreeDScene

    scene = ThreeDScene()
    # set_camera_orientation should not raise
    scene.set_camera_orientation(phi=75 * np.pi / 180, theta=-45 * np.pi / 180)

    # Verify camera params were set
    cam = scene.renderer.camera
    assert np.isclose(cam.phi_tracker.get_value(), 75 * np.pi / 180, atol=1e-10)
    assert np.isclose(cam.theta_tracker.get_value(), -45 * np.pi / 180, atol=1e-10)


def test_3d_geometry_tessellation():
    """Tessellation should produce 3-component vertices."""
    from manim_metal.utils import vmobject_to_stroke_quads, vmobject_to_triangles

    # Create a simple square path (4 cubic Bézier curves, 16 control points)
    # Each curve: P0, P1, P2, P3 (4 points × 3 coords)
    points = np.array(
        [
            # Curve 1: bottom edge
            [-1, -1, 0],
            [-0.5, -1, 0],
            [0.5, -1, 0],
            [1, -1, 0],
            # Curve 2: right edge
            [1, -1, 0],
            [1, -0.5, 0],
            [1, 0.5, 0],
            [1, 1, 0],
            # Curve 3: top edge
            [1, 1, 0],
            [0.5, 1, 0],
            [-0.5, 1, 0],
            [-1, 1, 0],
            # Curve 4: left edge
            [-1, 1, 0],
            [-1, 0.5, 0],
            [-1, -0.5, 0],
            [-1, -1, 0],
        ],
        dtype=np.float64,
    )

    tris = vmobject_to_triangles(points)
    assert tris.shape[1] == 3  # 3 components per vertex
    assert len(tris) > 0

    quads = vmobject_to_stroke_quads(points, stroke_width=0.04)
    assert quads.shape[1] == 3  # 3 components per vertex
    assert len(quads) > 0


def test_3d_geometry_preserves_z():
    """Tessellation should preserve z coordinates from input points."""
    from manim_metal.utils import vmobject_to_triangles

    # Square at z=2.0
    points = np.array(
        [
            [-1, -1, 2],
            [-0.5, -1, 2],
            [0.5, -1, 2],
            [1, -1, 2],
            [1, -1, 2],
            [1, -0.5, 2],
            [1, 0.5, 2],
            [1, 1, 2],
            [1, 1, 2],
            [0.5, 1, 2],
            [-0.5, 1, 2],
            [-1, 1, 2],
            [-1, 1, 2],
            [-1, 0.5, 2],
            [-1, -0.5, 2],
            [-1, -1, 2],
        ],
        dtype=np.float64,
    )

    tris = vmobject_to_triangles(points)
    # All z values should be approximately 2.0
    assert np.allclose(tris[:, 2], 2.0, atol=0.01)


def test_render_3d_circle(metal_config):
    """Render a Circle with 3D camera orientation and verify output."""
    from manim import Circle
    from manim.scene.three_d_scene import ThreeDScene

    from manim_metal.renderer import MetalRenderer

    scene = ThreeDScene()
    assert isinstance(scene.renderer, MetalRenderer)

    circle = Circle()
    scene.add(circle)
    scene.set_camera_orientation(phi=60 * np.pi / 180, theta=-45 * np.pi / 180)
    scene.renderer.update_frame(scene)
    frame = scene.renderer.get_frame()

    assert frame.shape == (
        scene.renderer.camera.pixel_height,
        scene.renderer.camera.pixel_width,
        4,
    )
    # Frame should have some rendered pixels
    bg = scene.renderer.camera.background
    diff = np.any(frame != bg, axis=2)
    assert diff.any(), "3D rendered frame should contain visible circle"
