"""Integration tests for Phase 2: Blinn-Phong lighting on 3D surfaces.

D3: Render a lit Sphere and verify pixels + lighting detection.
D4: Mixed lit + unlit scene (Surface + Circle) with correct pipeline selection.
D5: Regression test — 2D rendering is unchanged (no lighting applied).
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest


@pytest.fixture
def metal_config():
    """Set config to use Metal renderer, restore after test."""
    from manim import config

    original = config.renderer
    config.renderer = "metal"
    yield config
    config.renderer = original


# =====================================================================
# D3: Integration test — render a lit Sphere
# =====================================================================


class TestLitSphereRendering:
    """Verify that a Sphere renders with Blinn-Phong lighting pipeline."""

    def test_needs_lighting_detects_sphere(self, metal_config):
        """_needs_lighting should return True for Sphere submobjects."""
        from manim.mobject.three_d.three_dimensions import Sphere

        from manim_metal.metal_camera import _needs_lighting

        sphere = Sphere()
        # Sphere is composed of VMobject submobs (Surface patches)
        family = sphere.get_family()
        # At least the Sphere itself or its Surface parent should be detected
        lit_count = sum(1 for m in family if _needs_lighting(m))
        assert lit_count > 0, (
            f"Expected at least one family member to need lighting, "
            f"but none did out of {len(family)} members"
        )

    def test_needs_lighting_detects_surface(self, metal_config):
        """_needs_lighting should return True for Surface instances."""
        from manim.mobject.three_d.three_dimensions import Surface

        from manim_metal.metal_camera import _needs_lighting

        surface = Surface(
            lambda u, v: np.array([u, v, u * v]),
            u_range=[-1, 1],
            v_range=[-1, 1],
        )
        family = surface.get_family()
        lit_count = sum(1 for m in family if _needs_lighting(m))
        assert lit_count > 0, "Surface should be detected as needing lighting"

    def test_needs_lighting_detects_shade_in_3d(self, metal_config):
        """_needs_lighting should return True for objects with shade_in_3d=True."""
        from manim import Circle

        from manim_metal.metal_camera import _needs_lighting

        circle = Circle()
        assert not _needs_lighting(circle), "Circle should NOT need lighting by default"

        circle.shade_in_3d = True
        assert _needs_lighting(circle), "Circle with shade_in_3d=True SHOULD need lighting"

    def test_render_sphere_produces_pixels(self, metal_config):
        """Rendering a Sphere with a 3D camera should produce visible pixels."""
        from manim.mobject.three_d.three_dimensions import Sphere
        from manim.mobject.types.vectorized_mobject import VMobject

        from manim_metal.metal_camera import MetalCamera

        camera = MetalCamera(pixel_width=320, pixel_height=240)
        camera.set_phi(np.pi / 3)
        camera.set_theta(-np.pi / 4)

        sphere = Sphere()

        # Get all VMobject family members (the actual renderable submobs)
        family = [m for m in sphere.get_family() if isinstance(m, VMobject)]
        assert len(family) > 0, "Sphere should have VMobject family members"

        camera.capture_mobjects(family)
        frame = camera.pixel_array

        # Verify frame dimensions
        assert frame.shape == (240, 320, 4)

        # Check that something rendered (not just background)
        bg = camera.background
        diff = np.any(frame != bg, axis=2)
        rendered_pixels = np.sum(diff)
        assert rendered_pixels > 100, (
            f"Expected visible sphere pixels, but only {rendered_pixels} "
            f"pixels differ from background"
        )

    def test_lit_uniforms_are_packed(self, metal_config):
        """When rendering lit objects, uniform data should include lighting params."""
        from manim_metal.metal_camera import MetalCamera

        camera = MetalCamera(pixel_width=64, pixel_height=64)
        camera.set_phi(0.5)
        camera.set_theta(-0.3)
        camera.reset_rotation_matrix()

        color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Without lighting
        buf_unlit = camera._make_uniform_data(color, use_lighting=False)
        assert buf_unlit.view(np.uint32)[41] == 0, "use_lighting should be 0 for unlit"

        # With lighting
        buf_lit = camera._make_uniform_data(color, use_lighting=True)
        assert buf_lit.view(np.uint32)[41] == 1, "use_lighting should be 1 for lit"

        # Check lighting params are packed (Cairo-matching defaults:
        # ambient=0.5, diffuse=0.5, specular=0.0, shininess=3.0)
        assert buf_lit[42] == pytest.approx(0.5, abs=1e-6)  # ambient
        assert buf_lit[43] == pytest.approx(0.5, abs=1e-6)  # diffuse (Cairo intensity)
        assert buf_lit[52] == pytest.approx(0.0, abs=1e-6)  # specular (unused)
        assert buf_lit[53] == pytest.approx(3.0, abs=1e-6)  # shininess (Cairo exponent)


# =====================================================================
# D4: Integration test — mixed lit + unlit scene
# =====================================================================


class TestMixedLitUnlitScene:
    """Verify that a scene with both lit (3D surface) and unlit (Circle)
    objects renders correctly with the right pipeline for each."""

    def test_pipeline_selection(self, metal_config):
        """Lit objects should use _OP_FILL_COVER_LIT, unlit should use _OP_FILL_COVER."""
        from manim import Circle
        from manim.mobject.three_d.three_dimensions import Sphere

        from manim_metal.metal_camera import _needs_lighting

        sphere = Sphere()
        circle = Circle()

        # Check detection
        sphere_family = sphere.get_family()
        circle_family = circle.get_family()

        sphere_lit = any(_needs_lighting(m) for m in sphere_family)
        circle_lit = any(_needs_lighting(m) for m in circle_family)

        assert sphere_lit, "Sphere should be detected as lit"
        assert not circle_lit, "Circle should NOT be detected as lit"

    def test_mixed_scene_renders(self, metal_config):
        """A scene with both a Sphere and a Circle should render without errors."""
        from manim import Circle
        from manim.mobject.three_d.three_dimensions import Sphere
        from manim.mobject.types.vectorized_mobject import VMobject

        from manim_metal.metal_camera import MetalCamera

        camera = MetalCamera(pixel_width=320, pixel_height=240)
        camera.set_phi(np.pi / 4)
        camera.set_theta(-np.pi / 6)

        sphere = Sphere()
        circle = Circle().shift(np.array([2.0, 0.0, 0.0]))

        # Combine all family members
        all_mobs = [
            m
            for parent in [sphere, circle]
            for m in parent.get_family()
            if isinstance(m, VMobject) and len(m.points) >= 4
        ]

        # Should not raise
        camera.capture_mobjects(all_mobs)
        frame = camera.pixel_array

        # Both should contribute pixels
        bg = camera.background
        diff = np.any(frame != bg, axis=2)
        rendered_pixels = np.sum(diff)
        assert rendered_pixels > 100, (
            f"Expected rendered pixels from both objects, got {rendered_pixels}"
        )

    def test_z_ordering_preserved(self, metal_config):
        """Objects should render in the order they appear in the mobject list."""
        from manim import Circle, Square
        from manim.mobject.types.vectorized_mobject import VMobject

        from manim_metal.metal_camera import MetalCamera

        camera = MetalCamera(pixel_width=160, pixel_height=120)

        # Two overlapping 2D shapes with different colors
        sq = Square(color="#FF0000", fill_opacity=1.0).scale(0.5)
        ci = Circle(color="#0000FF", fill_opacity=1.0).scale(0.5)

        # Square first, circle on top (circle should occlude square center)
        all_mobs = []
        for parent in [sq, ci]:
            for m in parent.get_family():
                if isinstance(m, VMobject) and len(m.points) >= 4:
                    all_mobs.append(m)

        camera.capture_mobjects(all_mobs)
        frame_sq_then_ci = camera.pixel_array.copy()

        # Reverse order: circle first, square on top
        camera.reset()
        all_mobs_rev = []
        for parent in [ci, sq]:
            for m in parent.get_family():
                if isinstance(m, VMobject) and len(m.points) >= 4:
                    all_mobs_rev.append(m)

        camera.capture_mobjects(all_mobs_rev)
        frame_ci_then_sq = camera.pixel_array.copy()

        # Center pixel should differ between the two orderings
        cy, cx = 60, 80
        center_region = frame_sq_then_ci[cy - 5 : cy + 5, cx - 5 : cx + 5, :3]
        center_region_rev = frame_ci_then_sq[cy - 5 : cy + 5, cx - 5 : cx + 5, :3]

        # The frames should differ in the center region where overlap occurs
        # (one has blue on top, the other has red on top)
        center_diff = np.abs(center_region.astype(np.int16) - center_region_rev.astype(np.int16))
        assert np.max(center_diff) > 30, (
            "Z-ordering should produce different center pixels when order is reversed"
        )


# =====================================================================
# D5: Regression test — 2D rendering is unchanged
# =====================================================================


class TestRegression2D:
    """Verify that 2D scenes render correctly with no lighting applied."""

    def test_2d_circle_no_lighting(self, metal_config):
        """A 2D Circle should render without any lighting pipeline involvement."""
        from manim import Circle
        from manim.mobject.types.vectorized_mobject import VMobject

        from manim_metal.metal_camera import MetalCamera, _needs_lighting

        camera = MetalCamera(pixel_width=160, pixel_height=120)

        circle = Circle(fill_opacity=1.0)
        family = [m for m in circle.get_family() if isinstance(m, VMobject)]

        # No member should trigger lighting
        for m in family:
            assert not _needs_lighting(m), f"{type(m).__name__} should not need lighting"

        # Render should succeed
        camera.capture_mobjects(family)
        frame = camera.pixel_array

        bg = camera.background
        diff = np.any(frame != bg, axis=2)
        assert diff.any(), "2D Circle should produce visible pixels"

    def test_2d_square_renders_correctly(self, metal_config):
        """A filled 2D Square should render with correct dimensions."""
        from manim import Square
        from manim.mobject.types.vectorized_mobject import VMobject

        from manim_metal.metal_camera import MetalCamera

        camera = MetalCamera(pixel_width=160, pixel_height=120)

        square = Square(fill_opacity=1.0, color="#FF0000")
        family = [m for m in square.get_family() if isinstance(m, VMobject)]

        camera.capture_mobjects(family)
        frame = camera.pixel_array

        # Should have rendered something
        bg = camera.background
        diff = np.any(frame != bg, axis=2)
        rendered_count = np.sum(diff)
        assert rendered_count > 50, f"Expected rendered square pixels, got {rendered_count}"

    def test_2d_uniforms_have_no_lighting(self, metal_config):
        """2D uniform buffers should have use_lighting=0 and zero lighting params."""
        from manim_metal.metal_camera import MetalCamera

        camera = MetalCamera(pixel_width=64, pixel_height=64)

        color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        buf = camera._make_uniform_data(color, use_lighting=False)

        # is_3d should be 0
        assert buf.view(np.uint32)[40] == 0

        # use_lighting should be 0
        assert buf.view(np.uint32)[41] == 0

        # Lighting params should be zero
        assert buf[42] == 0.0  # ambient
        assert buf[43] == 0.0  # diffuse
        npt.assert_array_equal(buf[44:47], [0.0, 0.0, 0.0])  # light_position
        npt.assert_array_equal(buf[48:51], [0.0, 0.0, 0.0])  # light_color
        assert buf[52] == 0.0  # specular
        assert buf[53] == 0.0  # shininess

    def test_2d_scene_via_renderer(self, metal_config):
        """A full Scene with basic shapes should work identically to before."""
        from manim import Circle, Scene, Square

        from manim_metal.renderer import MetalRenderer

        scene = Scene()
        assert isinstance(scene.renderer, MetalRenderer)

        scene.add(Square(fill_opacity=0.5))
        scene.add(Circle(fill_opacity=0.5))

        scene.renderer.update_frame(scene)
        frame = scene.renderer.get_frame()

        assert frame.shape[2] == 4  # RGBA
        bg = scene.renderer.camera.background
        diff = np.any(frame != bg, axis=2)
        assert diff.any(), "2D scene should produce visible pixels"

    def test_existing_render_circle_still_works(self, metal_config):
        """Reproduce the existing test_render_circle to verify no regression."""
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
        bg = scene.renderer.camera.background
        diff = np.any(frame != bg, axis=2)
        assert diff.any(), "Frame should contain rendered circle pixels different from background"

    def test_existing_3d_circle_still_works(self, metal_config):
        """Reproduce the existing test_render_3d_circle to verify no regression."""
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
        bg = scene.renderer.camera.background
        diff = np.any(frame != bg, axis=2)
        assert diff.any(), "3D rendered frame should contain visible circle"


# =====================================================================
# Supplementary: Interleaving and normal computation integration
# =====================================================================


class TestLitVertexInterleaving:
    """Verify the LitVertex interleaving used by the lit pipeline."""

    def test_interleave_pos_normal_shape(self):
        """_interleave_pos_normal should produce (N, 6) float32."""
        from manim_metal.metal_camera import _interleave_pos_normal

        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        result = _interleave_pos_normal(positions, normals)

        assert result.shape == (2, 6)
        assert result.dtype == np.float32
        # First vertex: [px, py, pz, nx, ny, nz]
        npt.assert_array_equal(result[0], [1.0, 2.0, 3.0, 0.0, 0.0, 1.0])
        npt.assert_array_equal(result[1], [4.0, 5.0, 6.0, 0.0, 1.0, 0.0])

    def test_interleave_byte_size_is_24_per_vertex(self):
        """Each LitVertex should be 24 bytes (matches GPU struct)."""
        from manim_metal.metal_camera import _interleave_pos_normal

        positions = np.zeros((10, 3), dtype=np.float32)
        normals = np.zeros((10, 3), dtype=np.float32)
        result = _interleave_pos_normal(positions, normals)

        # 10 vertices * 6 floats * 4 bytes = 240 bytes
        # Per vertex: 6 * 4 = 24 bytes
        assert result.nbytes == 10 * 24

    def test_compute_face_normals_integration(self, metal_config):
        """compute_face_normals should produce unit normals for flat geometry."""
        from manim_metal.utils import compute_face_normals

        # Two triangles in XY plane (z=0)
        triangles = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
            ],
            dtype=np.float32,
        )
        normals = compute_face_normals(triangles)

        assert normals.shape == (6, 3)
        # All normals should point in +Z or -Z for XY-plane geometry
        for n in normals:
            assert abs(abs(n[2]) - 1.0) < 1e-5, f"Expected unit Z normal, got {n}"

    def test_batch_tessellate_with_normals(self, metal_config):
        """batch_tessellate(compute_normals=True) should return 4-tuples."""
        from manim_metal.utils import batch_tessellate

        # Simple square path
        points = np.array(
            [
                [-1, -1, 0],
                [-0.5, -1, 0],
                [0.5, -1, 0],
                [1, -1, 0],
                [1, -1, 0],
                [1, -0.5, 0],
                [1, 0.5, 0],
                [1, 1, 0],
                [1, 1, 0],
                [0.5, 1, 0],
                [-0.5, 1, 0],
                [-1, 1, 0],
                [-1, 1, 0],
                [-1, 0.5, 0],
                [-1, -0.5, 0],
                [-1, -1, 0],
            ],
            dtype=np.float64,
        )

        results = batch_tessellate([(points, 0.04)], compute_normals=True)

        assert len(results) == 1
        fill_tris, fill_normals, stroke_quads, stroke_normals = results[0]

        # Fill
        assert fill_tris.shape[1] == 3
        assert fill_normals.shape[1] == 3
        assert fill_tris.shape[0] == fill_normals.shape[0]

        # Stroke
        assert stroke_quads.shape[1] == 3
        assert stroke_normals.shape[1] == 3
        assert stroke_quads.shape[0] == stroke_normals.shape[0]

        # All normals should be unit length
        fill_mags = np.linalg.norm(fill_normals, axis=1)
        npt.assert_allclose(fill_mags, 1.0, atol=1e-5)

        stroke_mags = np.linalg.norm(stroke_normals, axis=1)
        npt.assert_allclose(stroke_mags, 1.0, atol=1e-5)
