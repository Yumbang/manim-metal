"""Unit tests for Phase 2 normal computation.

Tests are organized into two sections:
  1. Pure math tests — validate the cross-product / normalization logic that
     any normal-computation implementation must satisfy.  These run NOW.
  2. API integration tests — verify that the actual normal-computation functions
     (once implemented) produce correct normals for known Manim mobjects.
     These are marked ``@pytest.mark.skip`` until the implementation lands.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

# ---------------------------------------------------------------------------
# Helper: pure-math normal computation (reference implementation)
# ---------------------------------------------------------------------------


def _triangle_face_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute the unit face normal for a triangle (v0, v1, v2) via cross product.

    Returns the normalized cross product of (v1 - v0) x (v2 - v0).
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    n = np.cross(edge1, edge2)
    mag = np.linalg.norm(n)
    if mag < 1e-12:
        return np.array([0.0, 0.0, 0.0])
    return n / mag


def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector, returning zero if magnitude is near-zero."""
    mag = np.linalg.norm(v)
    if mag < 1e-12:
        return np.zeros_like(v)
    return v / mag


# =====================================================================
# Section 1: Pure math tests (run now, no implementation dependency)
# =====================================================================


class TestTriangleFaceNormal:
    """Verify face-normal math against known triangle configurations."""

    def test_xy_plane_triangle_normal_is_z_up(self):
        """A CCW triangle in the XY plane should have normal (0, 0, 1)."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        normal = _triangle_face_normal(v0, v1, v2)
        npt.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-10)

    def test_cw_triangle_normal_is_z_down(self):
        """A CW triangle in the XY plane should have normal (0, 0, -1)."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([0.0, 1.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        normal = _triangle_face_normal(v0, v1, v2)
        npt.assert_allclose(normal, [0.0, 0.0, -1.0], atol=1e-10)

    def test_xz_plane_triangle(self):
        """A CCW triangle in the XZ plane should have normal (0, -1, 0)."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 0.0, 1.0])
        normal = _triangle_face_normal(v0, v1, v2)
        npt.assert_allclose(normal, [0.0, -1.0, 0.0], atol=1e-10)

    def test_yz_plane_triangle(self):
        """A CCW triangle in the YZ plane should have normal (1, 0, 0)."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([0.0, 1.0, 0.0])
        v2 = np.array([0.0, 0.0, 1.0])
        normal = _triangle_face_normal(v0, v1, v2)
        npt.assert_allclose(normal, [1.0, 0.0, 0.0], atol=1e-10)

    def test_arbitrary_tilted_triangle(self):
        """An arbitrary tilted triangle should have a unit-length normal."""
        v0 = np.array([1.0, 2.0, 3.0])
        v1 = np.array([4.0, 5.0, 6.0])
        v2 = np.array([2.0, 8.0, 1.0])
        normal = _triangle_face_normal(v0, v1, v2)
        npt.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-10)

    def test_degenerate_triangle_returns_zero(self):
        """Collinear vertices should produce a zero normal (degenerate case)."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([2.0, 0.0, 0.0])
        normal = _triangle_face_normal(v0, v1, v2)
        npt.assert_allclose(normal, [0.0, 0.0, 0.0], atol=1e-10)

    def test_face_normal_perpendicular_to_edges(self):
        """The face normal must be perpendicular to both triangle edges."""
        v0 = np.array([1.0, 0.0, 0.5])
        v1 = np.array([0.0, 2.0, 0.0])
        v2 = np.array([3.0, 1.0, -1.0])
        normal = _triangle_face_normal(v0, v1, v2)
        edge1 = v1 - v0
        edge2 = v2 - v0
        npt.assert_allclose(np.dot(normal, edge1), 0.0, atol=1e-10)
        npt.assert_allclose(np.dot(normal, edge2), 0.0, atol=1e-10)


class TestSphereNormalMath:
    """Verify sphere normal property: normal at surface point = normalize(point - center)."""

    @pytest.mark.parametrize(
        "point",
        [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, -1.0]),
        ],
        ids=["x+", "y+", "z+", "x-", "y-", "z-"],
    )
    def test_unit_sphere_axis_points(self, point):
        """On a unit sphere at origin, normal at axis points = the point itself."""
        center = np.array([0.0, 0.0, 0.0])
        expected_normal = _normalize(point - center)
        npt.assert_allclose(expected_normal, point, atol=1e-10)

    def test_unit_sphere_diagonal_point(self):
        """On a unit sphere, normal at (1/sqrt3, 1/sqrt3, 1/sqrt3) = normalized point."""
        s = 1.0 / np.sqrt(3.0)
        point = np.array([s, s, s])
        normal = _normalize(point)
        npt.assert_allclose(normal, point, atol=1e-10)
        npt.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-10)

    def test_offset_sphere(self):
        """For a sphere centered at (2, 3, 4), normal = normalize(point - center)."""
        center = np.array([2.0, 3.0, 4.0])
        radius = 1.5
        # Point on the sphere surface along +x
        point = center + np.array([radius, 0.0, 0.0])
        expected = np.array([1.0, 0.0, 0.0])
        normal = _normalize(point - center)
        npt.assert_allclose(normal, expected, atol=1e-10)

    def test_many_sphere_normals_are_unit_length(self):
        """Sample 100 random points on a unit sphere; all normals should be unit length."""
        rng = np.random.default_rng(42)
        # Generate random unit vectors via normalize of normal distribution
        raw = rng.standard_normal((100, 3))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        points = raw / norms  # all on unit sphere surface
        for p in points:
            normal = _normalize(p)
            npt.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-10)


class TestCylinderNormalMath:
    """Verify cylinder side normal property: radial direction in XY, zero Z."""

    @pytest.mark.parametrize(
        "theta",
        [0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi - 0.01],
    )
    def test_vertical_cylinder_side_normal(self, theta):
        """For a vertical cylinder (axis along Z), side normals should be (cos(t), sin(t), 0)."""
        radius = 1.0
        # Point on cylinder surface
        point = np.array([radius * np.cos(theta), radius * np.sin(theta), 5.0])
        # Center of the axis at the same height
        axis_point = np.array([0.0, 0.0, 5.0])
        # Radial direction: project out the Z component
        radial = point - axis_point
        radial[2] = 0.0  # remove Z
        normal = _normalize(radial)
        expected = np.array([np.cos(theta), np.sin(theta), 0.0])
        npt.assert_allclose(normal, expected, atol=1e-10)

    def test_cylinder_normals_have_no_z_component(self):
        """Cylinder side normals should have exactly zero Z component."""
        for theta in np.linspace(0, 2 * np.pi, 20, endpoint=False):
            point = np.array([np.cos(theta), np.sin(theta), 3.0])
            radial = point.copy()
            radial[2] = 0.0
            normal = _normalize(radial)
            assert abs(normal[2]) < 1e-15, f"Z component should be 0, got {normal[2]}"

    def test_cylinder_normals_are_unit_length(self):
        """All cylinder side normals should have magnitude 1."""
        for theta in np.linspace(0, 2 * np.pi, 50, endpoint=False):
            point = np.array([2.0 * np.cos(theta), 2.0 * np.sin(theta), 0.0])
            radial = point.copy()
            radial[2] = 0.0
            normal = _normalize(radial)
            npt.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-10)


class TestNormalizationInvariants:
    """General invariants that any normal computation must satisfy."""

    def test_all_normals_unit_length_random_triangles(self):
        """Face normals of random non-degenerate triangles must all be unit length."""
        rng = np.random.default_rng(123)
        for _ in range(200):
            verts = rng.standard_normal((3, 3))
            normal = _triangle_face_normal(verts[0], verts[1], verts[2])
            mag = np.linalg.norm(normal)
            # Degenerate triangles (collinear) produce zero normal — skip those
            if mag > 0.5:
                npt.assert_allclose(mag, 1.0, atol=1e-10)

    def test_opposite_winding_flips_normal(self):
        """Reversing triangle winding should negate the face normal."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.5, 1.0, 0.3])
        n_ccw = _triangle_face_normal(v0, v1, v2)
        n_cw = _triangle_face_normal(v0, v2, v1)
        npt.assert_allclose(n_ccw, -n_cw, atol=1e-10)

    def test_translation_invariance(self):
        """Face normal should not change if the triangle is translated."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        offset = np.array([100.0, -50.0, 25.0])
        n_original = _triangle_face_normal(v0, v1, v2)
        n_translated = _triangle_face_normal(v0 + offset, v1 + offset, v2 + offset)
        npt.assert_allclose(n_original, n_translated, atol=1e-10)

    def test_uniform_scaling_invariance(self):
        """Face normal should not change under uniform positive scaling."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.5])
        scale = 17.3
        n_original = _triangle_face_normal(v0, v1, v2)
        n_scaled = _triangle_face_normal(v0 * scale, v1 * scale, v2 * scale)
        npt.assert_allclose(n_original, n_scaled, atol=1e-10)


# =====================================================================
# Section 2: API integration tests (skip until Phase 2 implementation)
# =====================================================================


@pytest.mark.skip(reason="Waiting for Phase 2: compute_face_normals not yet implemented")
class TestFlatSurfaceNormals:
    """Flat 2D VMobjects (z=0 everywhere) should have normals = (0, 0, 1)."""

    def test_square_normals(self):
        """All normals of a Square (flat in XY) should be (0, 0, 1)."""
        from manim import Square

        from manim_metal.utils import compute_face_normals

        sq = Square()
        normals = compute_face_normals(sq.points)
        assert normals.shape[1] == 3
        assert len(normals) > 0
        # All normals should point in +Z
        npt.assert_allclose(normals, np.array([0.0, 0.0, 1.0]), atol=1e-5)

    def test_circle_normals(self):
        """All normals of a Circle (flat in XY) should be (0, 0, 1)."""
        from manim import Circle

        from manim_metal.utils import compute_face_normals

        circle = Circle()
        normals = compute_face_normals(circle.points)
        assert normals.shape[1] == 3
        for n in normals:
            npt.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-5)


@pytest.mark.skip(reason="Waiting for Phase 2: Sphere normal computation not yet implemented")
class TestSphereNormals:
    """For a Sphere centered at origin, normal at any point = normalize(point)."""

    def test_sphere_normals_match_position(self):
        """Normals on a unit sphere should equal the normalized vertex positions."""
        from manim.mobject.three_d.three_dimensions import Sphere

        from manim_metal.utils import compute_vertex_normals

        sphere = Sphere(radius=1.0)
        # compute_vertex_normals should return (N, 3) array of unit normals
        positions, normals = compute_vertex_normals(sphere)
        assert positions.shape == normals.shape
        assert normals.shape[1] == 3

        for pos, nrm in zip(positions, normals):
            expected = _normalize(pos)
            # Allow some tolerance for tessellation discretization
            npt.assert_allclose(nrm, expected, atol=0.05)

    def test_sphere_normals_all_unit_length(self):
        """All normals on a sphere should be unit length."""
        from manim.mobject.three_d.three_dimensions import Sphere

        from manim_metal.utils import compute_vertex_normals

        sphere = Sphere(radius=2.0)
        _, normals = compute_vertex_normals(sphere)
        magnitudes = np.linalg.norm(normals, axis=1)
        npt.assert_allclose(magnitudes, 1.0, atol=1e-5)


@pytest.mark.skip(reason="Waiting for Phase 2: Cylinder normal computation not yet implemented")
class TestCylinderNormals:
    """For a vertical Cylinder, side normals = radial direction with zero Z."""

    def test_cylinder_side_normals_radial(self):
        """Side normals should point radially outward from the axis."""
        from manim.mobject.three_d.three_dimensions import Cylinder

        from manim_metal.utils import compute_vertex_normals

        cyl = Cylinder(radius=1.0, height=2.0)
        positions, normals = compute_vertex_normals(cyl)

        # Filter to side vertices (not caps): |z| < height/2 - epsilon
        side_mask = np.abs(positions[:, 2]) < 0.9
        side_pos = positions[side_mask]
        side_nrm = normals[side_mask]

        if len(side_pos) == 0:
            pytest.skip("No side vertices found — mesh structure may differ")

        for pos, nrm in zip(side_pos, side_nrm):
            # Expected: radial direction, no Z component
            expected_xy = _normalize(np.array([pos[0], pos[1], 0.0]))
            npt.assert_allclose(nrm[2], 0.0, atol=0.1)
            # XY direction should match
            nrm_xy = _normalize(np.array([nrm[0], nrm[1], 0.0]))
            npt.assert_allclose(nrm_xy, expected_xy, atol=0.1)

    def test_cylinder_normals_unit_length(self):
        """All cylinder normals should be unit length."""
        from manim.mobject.three_d.three_dimensions import Cylinder

        from manim_metal.utils import compute_vertex_normals

        cyl = Cylinder(radius=1.0, height=2.0)
        _, normals = compute_vertex_normals(cyl)
        magnitudes = np.linalg.norm(normals, axis=1)
        npt.assert_allclose(magnitudes, 1.0, atol=1e-5)


@pytest.mark.skip(reason="Waiting for Phase 2: tessellation with normals not yet implemented")
class TestTessellationWithNormals:
    """Verify that tessellation functions produce per-vertex normals alongside positions."""

    def test_vmobject_to_triangles_returns_normals(self):
        """vmobject_to_triangles should return both positions and normals when normals=True."""
        from manim_metal.utils import vmobject_to_triangles

        # Simple square path (16 control points for 4 cubic curves)
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

        positions, normals = vmobject_to_triangles(points, compute_normals=True)
        assert positions.shape[1] == 3
        assert normals.shape[1] == 3
        assert positions.shape[0] == normals.shape[0]
        # Flat geometry — all normals should be (0, 0, 1) or (0, 0, -1)
        z_components = np.abs(normals[:, 2])
        npt.assert_allclose(z_components, 1.0, atol=1e-5)


# =====================================================================
# Section 3: Global normal smoothing tests
# =====================================================================


class TestComputeGloballySmooothNormals:
    """Test compute_globally_smooth_normals across sub-mobject boundaries."""

    def test_all_normals_are_unit_length(self):
        """Every output normal must be unit length."""
        from manim_metal.utils import compute_globally_smooth_normals

        # Two XY-plane triangles sharing an edge
        tri_a = np.array(
            [[0, 0, 0], [1, 0, 0], [0.5, 1, 0]],
            dtype=np.float32,
        )
        tri_b = np.array(
            [[1, 0, 0], [2, 0, 0], [1.5, 1, 0]],
            dtype=np.float32,
        )
        result = compute_globally_smooth_normals([tri_a, tri_b])
        for normals in result:
            magnitudes = np.linalg.norm(normals, axis=1)
            npt.assert_allclose(magnitudes, 1.0, atol=1e-5)

    def test_coincident_vertices_get_same_normal(self):
        """Vertices at the same position across arrays must receive identical normals."""
        from manim_metal.utils import compute_globally_smooth_normals

        # Two triangles sharing vertex at (1, 0, 0), with slightly different face normals
        # Triangle A: in XY plane
        tri_a = np.array(
            [[0, 0, 0], [1, 0, 0], [0.5, 1, 0]],
            dtype=np.float32,
        )
        # Triangle B: tilted out of XY plane
        tri_b = np.array(
            [[1, 0, 0], [2, 0, 0], [1.5, 0.5, 0.5]],
            dtype=np.float32,
        )
        result = compute_globally_smooth_normals([tri_a, tri_b])

        # Vertex index 1 of tri_a = (1, 0, 0) should match vertex index 0 of tri_b
        normal_a_shared = result[0][1]
        normal_b_shared = result[1][0]
        npt.assert_allclose(normal_a_shared, normal_b_shared, atol=1e-6)

    def test_non_coincident_vertices_independent(self):
        """Vertices at different positions should not influence each other's normals."""
        from manim_metal.utils import compute_globally_smooth_normals

        # Two well-separated triangles in XY plane
        tri_a = np.array(
            [[0, 0, 0], [1, 0, 0], [0.5, 1, 0]],
            dtype=np.float32,
        )
        tri_b = np.array(
            [[10, 10, 0], [11, 10, 0], [10.5, 11, 0]],
            dtype=np.float32,
        )
        result = compute_globally_smooth_normals([tri_a, tri_b])

        # Both should have z-up normals since they're in XY plane
        for normals in result:
            npt.assert_allclose(np.abs(normals[:, 2]), 1.0, atol=1e-5)

    def test_empty_input(self):
        """Empty input list should return empty output list."""
        from manim_metal.utils import compute_globally_smooth_normals

        result = compute_globally_smooth_normals([])
        assert result == []

    def test_single_array(self):
        """Single array should still produce valid unit normals."""
        from manim_metal.utils import compute_globally_smooth_normals

        tri = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            dtype=np.float32,
        )
        result = compute_globally_smooth_normals([tri])
        assert len(result) == 1
        npt.assert_allclose(np.linalg.norm(result[0], axis=1), 1.0, atol=1e-5)

    def test_empty_arrays_in_list(self):
        """Empty arrays mixed with non-empty should be handled gracefully."""
        from manim_metal.utils import compute_globally_smooth_normals

        empty = np.empty((0, 3), dtype=np.float32)
        tri = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            dtype=np.float32,
        )
        result = compute_globally_smooth_normals([empty, tri, empty])
        assert len(result) == 3
        assert len(result[0]) == 0
        assert len(result[1]) == 3
        assert len(result[2]) == 0

    def test_output_count_matches_input(self):
        """Number of output arrays and their sizes must match input."""
        from manim_metal.utils import compute_globally_smooth_normals

        arrays = [
            np.random.default_rng(42).standard_normal((6, 3)).astype(np.float32),
            np.random.default_rng(43).standard_normal((9, 3)).astype(np.float32),
            np.random.default_rng(44).standard_normal((3, 3)).astype(np.float32),
        ]
        result = compute_globally_smooth_normals(arrays)
        assert len(result) == len(arrays)
        for inp, out in zip(arrays, result):
            assert out.shape == inp.shape

    def test_smooth_averaging_reduces_discontinuity(self):
        """Shared edge vertices should get averaged normals, reducing face normal spread."""
        from manim_metal.utils import compute_globally_smooth_normals

        # Two adjacent triangle fans sharing an edge vertex.
        # Fan A: centroid (0,0,0), vertices on a slightly tilted plane
        # Fan B: centroid (1,0,0), vertices on a differently tilted plane
        # The shared vertex at (0.5, 0.5, 0.1) should get an averaged normal.
        tri_a = np.array(
            [
                [0, 0, 0],
                [0.5, 0.5, 0.1],
                [1, 0, 0],
            ],
            dtype=np.float32,
        )
        tri_b = np.array(
            [
                [1, 0, 0],
                [0.5, 0.5, 0.1],
                [1, 1, 0.2],
            ],
            dtype=np.float32,
        )
        result = compute_globally_smooth_normals([tri_a, tri_b])

        # The shared vertex (0.5, 0.5, 0.1) at tri_a[1] and tri_b[1] should match
        npt.assert_allclose(result[0][1], result[1][1], atol=1e-6)

        # The shared vertex (1, 0, 0) at tri_a[2] and tri_b[0] should match
        npt.assert_allclose(result[0][2], result[1][0], atol=1e-6)
