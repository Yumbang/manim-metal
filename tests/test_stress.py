"""Stress tests and lighting verification for manim-metal.

Complements the existing test suite with:
  - Stress tests: many objects, mixed lit/unlit, overlapping geometry
  - Lit primitive verification: Sphere, Torus, Cylinder, Cone
  - Lighting parameter variation: ambient, diffuse, specular effects
  - Light position effects on pixel distribution
  - Performance: render time under budget
  - Z-ordering: lit + unlit objects at different depths
"""

from __future__ import annotations

import time

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def metal_config():
    """Set config to use Metal renderer, restore after test."""
    from manim import config

    original = config.renderer
    config.renderer = "metal"
    yield config
    config.renderer = original


@pytest.fixture
def camera_3d(metal_config):
    """Create a 3D MetalCamera with lighting defaults for testing."""
    import manim

    from manim_metal.metal_camera import MetalCamera

    cam = MetalCamera(
        pixel_width=320,
        pixel_height=240,
        frame_width=14.22,
        frame_height=8.0,
    )
    cam.set_phi(60 * manim.DEGREES)
    cam.set_theta(-45 * manim.DEGREES)
    # Lighting defaults
    cam.set_light_position(np.array([-5.0, 5.0, 10.0]))
    cam.set_light_color(np.array([1.0, 1.0, 1.0]))
    cam.set_ambient_strength(0.3)
    cam.set_diffuse_strength(0.7)
    cam.set_specular_strength(0.5)
    cam.set_shininess(32.0)
    return cam


@pytest.fixture
def camera_2d(metal_config):
    """Create a 2D MetalCamera for testing."""
    from manim_metal.metal_camera import MetalCamera

    cam = MetalCamera(
        pixel_width=320,
        pixel_height=240,
        frame_width=14.22,
        frame_height=8.0,
    )
    return cam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def render_and_count(cam, mobjects):
    """Render mobjects and return non-background pixel count.

    Uses the camera background as the reference; any pixel whose alpha
    channel is non-zero (or that differs from the background) is counted.
    """
    from manim.mobject.types.vectorized_mobject import VMobject

    # Flatten to renderable VMobject family members
    flat = []
    for m in mobjects:
        for sub in m.get_family():
            if isinstance(sub, VMobject) and len(sub.points) >= 4:
                flat.append(sub)

    cam.reset()
    cam.capture_mobjects(flat)
    frame = cam.pixel_array

    bg = cam.background
    diff = np.any(frame != bg, axis=2)
    return int(np.sum(diff))


def render_frame(cam, mobjects):
    """Render mobjects and return the full RGBA frame as a numpy array."""
    from manim.mobject.types.vectorized_mobject import VMobject

    flat = []
    for m in mobjects:
        for sub in m.get_family():
            if isinstance(sub, VMobject) and len(sub.points) >= 4:
                flat.append(sub)

    cam.reset()
    cam.capture_mobjects(flat)
    return cam.pixel_array.copy()


# =====================================================================
# Stress Tests
# =====================================================================


class TestStressRendering:
    """Stress tests with many objects to verify stability and correctness."""

    def test_many_spheres(self, camera_3d):
        """Render 20+ Spheres simultaneously at various positions."""
        import manim
        from manim.mobject.three_d.three_dimensions import Sphere

        spheres = []
        for i in range(24):
            s = Sphere(radius=0.4)
            # Arrange in a grid pattern
            row = i // 6
            col = i % 6
            x = (col - 2.5) * 1.5
            y = (row - 1.5) * 1.5
            z = (i % 3) * 0.5
            s.shift(np.array([x, y, z]))
            colors = [manim.BLUE, manim.RED, manim.GREEN, manim.YELLOW, manim.PURPLE, manim.ORANGE]
            s.set_color(colors[i % len(colors)])
            spheres.append(s)

        pixel_count = render_and_count(camera_3d, spheres)
        assert pixel_count > 100, (
            f"Expected >100 rendered pixels from 24 spheres, got {pixel_count}"
        )

    def test_mixed_lit_unlit(self, camera_3d):
        """Create 30 lit objects and 30 unlit objects, render all together."""
        import manim
        from manim import Circle, Square
        from manim.mobject.three_d.three_dimensions import Sphere, Torus

        lit_objects = []
        for i in range(15):
            s = Sphere(radius=0.3)
            s.shift(np.array([(i % 5 - 2) * 1.5, (i // 5 - 1) * 1.5, 0.0]))
            s.set_color(manim.BLUE)
            lit_objects.append(s)
        for i in range(15):
            t = Torus(major_radius=0.4, minor_radius=0.15)
            t.shift(np.array([(i % 5 - 2) * 1.5, (i // 5 - 1) * 1.5 + 0.5, 0.5]))
            t.set_color(manim.GREEN)
            lit_objects.append(t)

        unlit_objects = []
        for i in range(15):
            c = Circle(radius=0.3, fill_opacity=0.8)
            c.shift(np.array([(i % 5 - 2) * 1.5 + 0.3, (i // 5 - 1) * 1.5, 0.0]))
            c.set_color(manim.RED)
            unlit_objects.append(c)
        for i in range(15):
            sq = Square(side_length=0.5, fill_opacity=0.8)
            sq.shift(np.array([(i % 5 - 2) * 1.5 - 0.3, (i // 5 - 1) * 1.5, 0.0]))
            sq.set_color(manim.YELLOW)
            unlit_objects.append(sq)

        all_objects = lit_objects + unlit_objects
        pixel_count = render_and_count(camera_3d, all_objects)
        assert pixel_count > 100, f"Expected >100 pixels from 60 mixed objects, got {pixel_count}"

    def test_many_overlapping_2d(self, camera_2d):
        """Render many overlapping 2D shapes to stress the stencil/fill pipeline."""
        import manim
        from manim import Circle, Square, Triangle

        shapes = []
        for i in range(40):
            if i % 3 == 0:
                obj = Circle(radius=0.5 + (i % 5) * 0.1, fill_opacity=0.6)
            elif i % 3 == 1:
                obj = Square(side_length=0.5 + (i % 4) * 0.15, fill_opacity=0.6)
            else:
                obj = Triangle(fill_opacity=0.6)
                obj.scale(0.5 + (i % 3) * 0.2)
            angle = i * 2 * np.pi / 40
            obj.shift(np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0]))
            colors = [manim.BLUE, manim.RED, manim.GREEN, manim.YELLOW]
            obj.set_color(colors[i % len(colors)])
            shapes.append(obj)

        pixel_count = render_and_count(camera_2d, shapes)
        assert pixel_count > 100, (
            f"Expected >100 pixels from 40 overlapping 2D shapes, got {pixel_count}"
        )


# =====================================================================
# Lit Primitives
# =====================================================================


class TestLitPrimitives:
    """Verify that each 3D primitive type renders with lighting."""

    def test_sphere_lit(self, camera_3d):
        """Sphere with lighting should produce visible pixels."""
        import manim
        from manim.mobject.three_d.three_dimensions import Sphere

        sphere = Sphere(radius=1.0)
        sphere.set_color(manim.BLUE)
        pixel_count = render_and_count(camera_3d, [sphere])
        assert pixel_count > 100, f"Expected >100 pixels for lit Sphere, got {pixel_count}"

    def test_torus_lit(self, camera_3d):
        """Torus with lighting should produce visible pixels."""
        import manim
        from manim.mobject.three_d.three_dimensions import Torus

        torus = Torus(major_radius=1.5, minor_radius=0.5)
        torus.set_color(manim.GREEN)
        pixel_count = render_and_count(camera_3d, [torus])
        assert pixel_count > 100, f"Expected >100 pixels for lit Torus, got {pixel_count}"

    def test_cylinder_lit(self, camera_3d):
        """Cylinder with lighting should produce visible pixels."""
        import manim
        from manim.mobject.three_d.three_dimensions import Cylinder

        cylinder = Cylinder(radius=0.5, height=2.0)
        cylinder.set_color(manim.RED)
        pixel_count = render_and_count(camera_3d, [cylinder])
        assert pixel_count > 100, f"Expected >100 pixels for lit Cylinder, got {pixel_count}"

    def test_cone_lit(self, camera_3d):
        """Cone with lighting should produce visible pixels."""
        import manim
        from manim.mobject.three_d.three_dimensions import Cone

        cone = Cone(base_radius=1.0, height=2.0)
        cone.set_color(manim.YELLOW)
        pixel_count = render_and_count(camera_3d, [cone])
        assert pixel_count > 100, f"Expected >100 pixels for lit Cone, got {pixel_count}"


# =====================================================================
# Lighting Parameter Effects
# =====================================================================


class TestLightingParams:
    """Verify that different lighting parameters produce different pixel outputs."""

    def _render_sphere_with_params(
        self, metal_config, ambient=0.2, diffuse=0.4, specular=0.5, shininess=32.0
    ):
        """Helper: render a sphere with given lighting params and return frame."""
        import manim
        from manim.mobject.three_d.three_dimensions import Sphere

        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera(
            pixel_width=320,
            pixel_height=240,
            frame_width=14.22,
            frame_height=8.0,
        )
        cam.set_phi(60 * manim.DEGREES)
        cam.set_theta(-45 * manim.DEGREES)
        cam.set_light_position(np.array([-5.0, 5.0, 10.0]))
        cam.set_light_color(np.array([1.0, 1.0, 1.0]))
        cam.set_ambient_strength(ambient)
        cam.set_diffuse_strength(diffuse)
        cam.set_specular_strength(specular)
        cam.set_shininess(shininess)

        sphere = Sphere(radius=1.0)
        sphere.set_color(manim.BLUE)
        return render_frame(cam, [sphere])

    def test_intensity_variation(self, metal_config):
        """Different intensity (diffuse) values should produce different brightness."""
        frame_low = self._render_sphere_with_params(metal_config, diffuse=0.1)
        frame_high = self._render_sphere_with_params(metal_config, diffuse=0.9)

        diff = np.abs(frame_low.astype(np.int16) - frame_high.astype(np.int16))
        max_diff = np.max(diff[..., :3])
        assert max_diff > 0, (
            "Different intensity (diffuse) values should produce pixel differences"
        )

    def test_diffuse_variation(self, metal_config):
        """Different diffuse strengths should produce different pixel distributions."""
        frame_low = self._render_sphere_with_params(metal_config, diffuse=0.1)
        frame_high = self._render_sphere_with_params(metal_config, diffuse=0.9)

        diff = np.abs(frame_low.astype(np.int16) - frame_high.astype(np.int16))
        max_diff = np.max(diff[..., :3])
        assert max_diff > 0, (
            "Different diffuse strengths should produce at least some pixel difference"
        )

    def test_exponent_variation(self, metal_config):
        """Different exponents (shininess) should change the falloff pattern."""
        frame_low = self._render_sphere_with_params(metal_config, shininess=1.0)
        frame_high = self._render_sphere_with_params(metal_config, shininess=8.0)

        diff = np.abs(frame_low.astype(np.int16) - frame_high.astype(np.int16))
        max_diff = np.max(diff[..., :3])
        assert max_diff > 0, (
            "Different exponents (shininess) should produce pixel differences"
        )

    def test_light_position_effect(self, metal_config):
        """Moving the light source should change the pixel distribution."""
        import manim
        from manim.mobject.three_d.three_dimensions import Sphere

        from manim_metal.metal_camera import MetalCamera

        def render_with_light_pos(pos):
            cam = MetalCamera(
                pixel_width=320,
                pixel_height=240,
                frame_width=14.22,
                frame_height=8.0,
            )
            cam.set_phi(60 * manim.DEGREES)
            cam.set_theta(-45 * manim.DEGREES)
            cam.set_light_position(np.array(pos, dtype=np.float64))
            cam.set_light_color(np.array([1.0, 1.0, 1.0]))
            cam.set_ambient_strength(0.2)
            cam.set_diffuse_strength(0.8)
            cam.set_specular_strength(0.5)
            cam.set_shininess(32.0)
            sphere = Sphere(radius=1.0)
            sphere.set_color(manim.BLUE)
            return render_frame(cam, [sphere])

        frame_left = render_with_light_pos([-10.0, 0.0, 5.0])
        frame_right = render_with_light_pos([10.0, 0.0, 5.0])

        diff = np.abs(frame_left.astype(np.int16) - frame_right.astype(np.int16))
        max_diff = np.max(diff[..., :3])
        assert max_diff > 0, (
            "Different light positions should produce different pixel distributions"
        )


# =====================================================================
# Performance
# =====================================================================


class TestPerformance:
    """Performance tests to ensure rendering stays within time budgets."""

    @pytest.mark.slow
    def test_lit_render_time(self, metal_config):
        """Rendering a lit scene at 960x540 should complete in under 5 seconds."""
        import manim
        from manim.mobject.three_d.three_dimensions import Sphere, Torus
        from manim.mobject.types.vectorized_mobject import VMobject

        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera(
            pixel_width=960,
            pixel_height=540,
            frame_width=14.22,
            frame_height=8.0,
        )
        cam.set_phi(60 * manim.DEGREES)
        cam.set_theta(-45 * manim.DEGREES)
        cam.set_light_position(np.array([-5.0, 5.0, 10.0]))
        cam.set_light_color(np.array([1.0, 1.0, 1.0]))
        cam.set_ambient_strength(0.3)
        cam.set_diffuse_strength(0.7)
        cam.set_specular_strength(0.5)
        cam.set_shininess(32.0)

        # Build a scene with several lit objects
        objects = []
        for i in range(5):
            s = Sphere(radius=0.5)
            s.shift(np.array([(i - 2) * 2.0, 0.0, 0.0]))
            s.set_color(manim.BLUE)
            objects.append(s)
        for i in range(3):
            t = Torus(major_radius=0.8, minor_radius=0.3)
            t.shift(np.array([(i - 1) * 3.0, 2.0, 0.0]))
            t.set_color(manim.GREEN)
            objects.append(t)

        # Flatten to renderable family members
        flat = []
        for m in objects:
            for sub in m.get_family():
                if isinstance(sub, VMobject) and len(sub.points) >= 4:
                    flat.append(sub)

        # Warm-up pass (shader compilation, buffer allocation)
        cam.reset()
        cam.capture_mobjects(flat)

        # Timed pass
        start = time.perf_counter()
        cam.reset()
        cam.capture_mobjects(flat)
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"Lit scene render at 960x540 took {elapsed:.3f}s, expected <5.0s"

        # Verify it actually rendered something
        bg = cam.background
        diff = np.any(cam.pixel_array != bg, axis=2)
        assert np.sum(diff) > 100, "Performance test scene should produce visible pixels"


# =====================================================================
# Z-ordering
# =====================================================================


class TestZOrdering:
    """Verify correct rendering of overlapping lit and unlit objects at different depths."""

    def test_lit_unlit_overlap(self, camera_3d):
        """Overlapping lit (Sphere) and unlit (Circle) objects should render correctly."""
        import manim
        from manim import Circle
        from manim.mobject.three_d.three_dimensions import Sphere

        # Sphere at origin (3D, lit)
        sphere = Sphere(radius=1.0)
        sphere.set_color(manim.BLUE)

        # Circle overlapping in front (2D, unlit)
        circle = Circle(radius=0.8, fill_opacity=0.9)
        circle.set_color(manim.RED)
        circle.shift(np.array([0.5, 0.0, 1.5]))

        # Should not crash regardless of order
        pixel_count = render_and_count(camera_3d, [sphere, circle])
        assert pixel_count > 100, (
            f"Expected >100 pixels from overlapping lit+unlit objects, got {pixel_count}"
        )

    def test_lit_objects_at_different_depths(self, camera_3d):
        """Multiple lit objects at different z depths should all render."""
        import manim
        from manim.mobject.three_d.three_dimensions import Sphere

        front = Sphere(radius=0.5)
        front.set_color(manim.RED)
        front.shift(np.array([0.0, 0.0, 2.0]))

        middle = Sphere(radius=0.5)
        middle.set_color(manim.GREEN)
        middle.shift(np.array([1.5, 0.0, 0.0]))

        back = Sphere(radius=0.5)
        back.set_color(manim.BLUE)
        back.shift(np.array([-1.5, 0.0, -2.0]))

        pixel_count = render_and_count(camera_3d, [front, middle, back])
        assert pixel_count > 100, (
            f"Expected >100 pixels from 3 spheres at different depths, got {pixel_count}"
        )

    def test_unlit_behind_lit(self, camera_3d):
        """An unlit Circle behind a lit Sphere should still render the visible parts."""
        import manim
        from manim import Square
        from manim.mobject.three_d.three_dimensions import Sphere

        sphere = Sphere(radius=0.6)
        sphere.set_color(manim.BLUE)

        # Large square behind the sphere
        sq = Square(side_length=4.0, fill_opacity=0.8)
        sq.set_color(manim.YELLOW)
        sq.shift(np.array([0.0, 0.0, -2.0]))

        pixel_count = render_and_count(camera_3d, [sq, sphere])
        assert pixel_count > 100, (
            f"Expected >100 pixels for sphere in front of square, got {pixel_count}"
        )
