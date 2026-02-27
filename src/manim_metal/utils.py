"""NumPy / Metal buffer conversion helpers and VMobject geometry utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

# ---------------------------------------------------------------------------
# Pre-computed Bernstein basis coefficients for n_samples=64
# These are constant — computing them once avoids repeated power ops.
# ---------------------------------------------------------------------------
N_CURVE_SAMPLES = 64

_t = np.linspace(0, 1, N_CURVE_SAMPLES, dtype=np.float64).reshape(
    N_CURVE_SAMPLES, 1, 1
)
_mt = 1.0 - _t
_B0 = _mt**3
_B1 = 3.0 * _mt**2 * _t
_B2 = 3.0 * _mt * _t**2
_B3 = _t**3

# Extra half-width added to stroke quads so the shader's fwidth-based
# alpha fade has room to smooth edges without visually thinning the stroke.
# Approximately 1 pixel at default frame (14.22 units / 1920 px).
_STROKE_FEATHER = 0.008


def _linearize_beziers_batch(
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Linearize all cubic Bézier curves in a single batched NumPy operation.

    Uses pre-computed Bernstein coefficients for speed.

    Parameters
    ----------
    points
        (n_curves*4, 3) array of 3D control points.

    Returns
    -------
    np.ndarray
        (n_curves * N_CURVE_SAMPLES, 3) array of linearized points.
    """
    n_curves = len(points) // 4
    if n_curves == 0:
        return np.empty((0, 3), dtype=np.float64)

    # Reshape to (n_curves, 4, 3) — P0, P1, P2, P3 per curve
    ctrl = points[: n_curves * 4].reshape(n_curves, 4, 3)

    # Vectorized cubic evaluation with pre-computed basis
    # Basis shape (N, 1, 1) broadcasts to (N, n_curves, 3)
    curve_pts = (
        _B0 * ctrl[:, 0] + _B1 * ctrl[:, 1] + _B2 * ctrl[:, 2] + _B3 * ctrl[:, 3]
    )

    # Transpose to (n_curves, N_CURVE_SAMPLES, 3), flatten
    return curve_pts.transpose(1, 0, 2).reshape(-1, 3)


def _dedup_polyline(polyline: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Remove consecutive duplicate points from a polyline."""
    if len(polyline) < 2:
        return polyline
    mask = np.ones(len(polyline), dtype=bool)
    mask[1:] = np.any(np.abs(polyline[1:] - polyline[:-1]) > 1e-8, axis=1)
    return polyline[mask]


def _polyline_to_fan(polyline: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
    """Build triangle fan from centroid for a single deduped polyline."""
    if len(polyline) < 3:
        return np.empty((0, 3), dtype=np.float32)
    centroid = polyline.mean(axis=0)
    n = len(polyline)
    triangles = np.empty((n, 3, 3), dtype=np.float32)
    triangles[:, 0, :] = centroid.astype(np.float32)
    triangles[:, 1, :] = polyline.astype(np.float32)
    triangles[:, 2, :] = np.roll(polyline, -1, axis=0).astype(np.float32)
    return triangles.reshape(-1, 3)


def _polyline_to_stroke(
    polyline: npt.NDArray[np.float64], half_w: float
) -> npt.NDArray[np.float32]:
    """Expand a deduped polyline into stroke quads.

    Normals are computed in the XY plane; z is preserved from the polyline
    vertices. This is correct for flat shapes in 3D space.

    A small feather is added to the half-width so the shader's fwidth-based
    alpha fade has room to produce smooth edges without thinning the stroke.
    """
    if len(polyline) < 2:
        return np.empty((0, 3), dtype=np.float32)

    a = polyline[:-1]
    b = polyline[1:]
    d = b - a
    # Compute 2D normals from XY components
    lengths = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)

    safe_lengths = np.where(lengths < 1e-10, 1.0, lengths)

    # Add feather for stroke edge anti-aliasing (~1 pixel at default zoom)
    expanded_hw = half_w + _STROKE_FEATHER

    # 3D normals: perpendicular in XY, z=0
    normals = np.zeros_like(d)
    normals[:, 0] = -d[:, 1] / safe_lengths * expanded_hw
    normals[:, 1] = d[:, 0] / safe_lengths * expanded_hw
    # normals[:, 2] stays 0 — expand in XY plane, preserve z from endpoints

    p0 = a + normals
    p1 = a - normals
    p2 = b + normals
    p3 = b - normals

    n_segments = len(a)
    quads = np.empty((n_segments, 6, 3), dtype=np.float32)
    quads[:, 0] = p0
    quads[:, 1] = p1
    quads[:, 2] = p2
    quads[:, 3] = p1
    quads[:, 4] = p3
    quads[:, 5] = p2

    degen_mask = lengths < 1e-10
    if np.any(degen_mask):
        quads[degen_mask] = a[degen_mask, np.newaxis, :].astype(np.float32)

    return quads.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Public API — single-object tessellation (used by geometry cache misses)
# ---------------------------------------------------------------------------


def vmobject_to_triangles(
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float32]:
    """Convert VMobject cubic Bézier points to triangle fan vertices for stencil fill."""
    if len(points) < 4:
        return np.empty((0, 3), dtype=np.float32)

    n_curves = len(points) // 4
    points_3d = points[: n_curves * 4, :3]
    polyline = _dedup_polyline(_linearize_beziers_batch(points_3d))
    return _polyline_to_fan(polyline)


def vmobject_to_stroke_quads(
    points: npt.NDArray[np.float64],
    stroke_width: float,
) -> npt.NDArray[np.float32]:
    """Convert VMobject Bézier points to thick-line quad strip for stroke rendering."""
    if len(points) < 4 or stroke_width <= 0:
        return np.empty((0, 3), dtype=np.float32)

    n_curves = len(points) // 4
    points_3d = points[: n_curves * 4, :3]
    polyline = _dedup_polyline(_linearize_beziers_batch(points_3d))
    return _polyline_to_stroke(polyline, stroke_width / 2.0)


# ---------------------------------------------------------------------------
# Batch tessellation — process many objects in one NumPy call
# ---------------------------------------------------------------------------


def batch_tessellate(
    items: list[tuple[npt.NDArray[np.float64], float | None]],
) -> list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]]:
    """Tessellate fill + stroke for many VMobjects in one fully-batched operation.

    All fan and stroke geometry is constructed in single vectorized NumPy
    operations across ALL objects simultaneously — no per-object Python loops.

    Parameters
    ----------
    items
        List of ``(points, stroke_width_or_None)`` per VMobject.
        *stroke_width* is the scene-unit stroke width (already scaled by 0.01),
        or ``None`` to skip stroke tessellation.

    Returns
    -------
    list[tuple]
        For each input: ``(fill_triangles, stroke_quads_or_None)``.
    """
    if not items:
        return []

    # --- Phase 1: Collect control points and curve counts ---
    curve_counts: list[int] = []
    all_ctrl: list[npt.NDArray[np.float64]] = []
    total_curves = 0

    for points, _ in items:
        n_curves = len(points) // 4 if len(points) >= 4 else 0
        curve_counts.append(n_curves)
        if n_curves > 0:
            all_ctrl.append(points[: n_curves * 4, :3])
            total_curves += n_curves

    if total_curves == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return [(empty, None if sw is None else empty) for _, sw in items]

    # --- Phase 2: Batched linearization (all curves at once) ---
    all_points_3d = np.concatenate(all_ctrl, axis=0)
    all_lin = _linearize_beziers_batch(all_points_3d)

    # Per-object polyline sizes and start offsets (non-zero objects only)
    nz_indices: list[int] = []
    nz_sizes_list: list[int] = []
    for i, nc in enumerate(curve_counts):
        if nc > 0:
            nz_indices.append(i)
            nz_sizes_list.append(nc * N_CURVE_SAMPLES)

    nz_sizes = np.array(nz_sizes_list, dtype=np.intp)
    N = len(all_lin)
    nz_starts = np.empty(len(nz_sizes), dtype=np.intp)
    nz_starts[0] = 0
    if len(nz_sizes) > 1:
        np.cumsum(nz_sizes[:-1], out=nz_starts[1:])

    # Object-boundary end indices
    nz_ends = nz_starts + nz_sizes - 1

    # --- Phase 3: Batched fan construction (all objects at once) ---
    # Compute per-object centroids via segment-reduce
    centroids = (
        np.add.reduceat(all_lin, nz_starts, axis=0) / nz_sizes[:, None]
    )
    # Expand centroids to match each point
    centroid_per_pt = np.repeat(centroids, nz_sizes, axis=0)

    # Next-point indices (wrapping within each object)
    next_idx = np.arange(N, dtype=np.intp) + 1
    next_idx[nz_ends] = nz_starts

    # Build all fan triangles in one shot
    all_fan = np.empty((N, 3, 3), dtype=np.float32)
    all_fan[:, 0, :] = centroid_per_pt.astype(np.float32)
    all_fan[:, 1, :] = all_lin.astype(np.float32)
    all_fan[:, 2, :] = all_lin[next_idx].astype(np.float32)
    all_fan_flat = all_fan.reshape(-1, 3)

    # Split positions for fans (cumulative vertex counts)
    fan_vert_counts = nz_sizes * 3
    fan_splits = np.cumsum(fan_vert_counts[:-1])

    # --- Phase 4: Batched stroke construction (all stroke objects at once) ---
    # Identify which non-zero objects need stroke and their half-widths
    stroke_nz_idx: list[int] = []
    stroke_half_ws: list[float] = []
    for j, orig_idx in enumerate(nz_indices):
        _, sw = items[orig_idx]
        if sw is not None:
            stroke_nz_idx.append(j)
            stroke_half_ws.append(sw / 2.0)

    # Pre-allocate per-nonzero-object stroke result slots
    stroke_results_per_nz: list[npt.NDArray[np.float32] | None] = [
        None
    ] * len(nz_indices)

    if stroke_nz_idx:
        # Collect all 'a' and 'b' endpoints for stroke segments across objects
        stroke_a_parts: list[npt.NDArray[np.float64]] = []
        stroke_b_parts: list[npt.NDArray[np.float64]] = []
        stroke_hw_parts: list[npt.NDArray[np.float64]] = []
        stroke_seg_counts: list[int] = []

        for k, j_nz in enumerate(stroke_nz_idx):
            start = int(nz_starts[j_nz])
            size = int(nz_sizes[j_nz])
            n_seg = size - 1
            if n_seg <= 0:
                stroke_seg_counts.append(0)
                continue
            stroke_seg_counts.append(n_seg)
            stroke_a_parts.append(all_lin[start : start + n_seg])
            stroke_b_parts.append(all_lin[start + 1 : start + 1 + n_seg])
            stroke_hw_parts.append(np.full(n_seg, stroke_half_ws[k]))

        if stroke_a_parts:
            # Concatenate ALL stroke segments from all objects
            sa = np.concatenate(stroke_a_parts, axis=0)
            sb = np.concatenate(stroke_b_parts, axis=0)
            hw = np.concatenate(stroke_hw_parts, axis=0)

            # Vectorized stroke expansion — normals in XY plane, z preserved
            d = sb - sa
            lengths = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
            safe_len = np.where(lengths < 1e-10, 1.0, lengths)

            # Add feather for stroke edge anti-aliasing
            expanded_hw = hw + _STROKE_FEATHER

            normals = np.zeros_like(d)
            normals[:, 0] = -d[:, 1] / safe_len * expanded_hw
            normals[:, 1] = d[:, 0] / safe_len * expanded_hw
            # normals[:, 2] stays 0

            p0 = (sa + normals).astype(np.float32)
            p1 = (sa - normals).astype(np.float32)
            p2 = (sb + normals).astype(np.float32)
            p3 = (sb - normals).astype(np.float32)

            n_seg_total = len(sa)
            quads = np.empty((n_seg_total, 6, 3), dtype=np.float32)
            quads[:, 0] = p0
            quads[:, 1] = p1
            quads[:, 2] = p2
            quads[:, 3] = p1
            quads[:, 4] = p3
            quads[:, 5] = p2

            degen = lengths < 1e-10
            if np.any(degen):
                quads[degen] = sa[degen, np.newaxis, :].astype(np.float32)

            # Split stroke quads back per stroke object
            all_quads_flat = quads.reshape(-1, 3)
            seg_vert_counts = np.array(stroke_seg_counts)
            quad_splits = np.cumsum(seg_vert_counts[:-1] * 6)
            stroke_chunks = np.split(all_quads_flat, quad_splits)

            chunk_idx = 0
            for k, j_nz in enumerate(stroke_nz_idx):
                if stroke_seg_counts[k] > 0:
                    stroke_results_per_nz[j_nz] = stroke_chunks[chunk_idx]
                    chunk_idx += 1

    # --- Phase 5: Assemble results per input object ---
    # Split fan triangles per non-zero object
    fan_chunks = np.split(all_fan_flat, fan_splits)

    results: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]] = []
    nz_pos = 0  # position in nz_indices / fan_chunks / stroke_results_per_nz

    for i, (_, stroke_width) in enumerate(items):
        nc = curve_counts[i]
        if nc == 0:
            empty = np.empty((0, 3), dtype=np.float32)
            results.append((empty, None if stroke_width is None else empty))
            continue

        fill_tris = fan_chunks[nz_pos]
        sq = stroke_results_per_nz[nz_pos]
        if stroke_width is not None and sq is None:
            sq = np.empty((0, 3), dtype=np.float32)

        results.append((fill_tris, sq))
        nz_pos += 1

    return results


def build_world_to_ndc_matrix(
    frame_width: float,
    frame_height: float,
    frame_center_x: float = 0.0,
    frame_center_y: float = 0.0,
) -> npt.NDArray[np.float32]:
    """Build a 4x4 orthographic projection matrix: manim world coords -> Metal NDC [-1,1].

    Metal NDC: x right, y up, z into screen [0,1].

    Parameters
    ----------
    frame_width, frame_height
        Visible area in manim world units.
    frame_center_x, frame_center_y
        Center of the frame in world coords (default origin).

    Returns
    -------
    np.ndarray
        4x4 float32 column-major matrix suitable for Metal uniform buffer.
    """
    sx = 2.0 / frame_width
    sy = 2.0 / frame_height
    tx = -frame_center_x * sx
    ty = -frame_center_y * sy

    # Column-major storage for Metal (which expects column-major float4x4)
    return np.array(
        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [tx, ty, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def build_rotation_matrix(
    phi: float, theta: float, gamma: float
) -> npt.NDArray[np.float64]:
    """Build a 3x3 rotation matrix matching ThreeDCamera.generate_rotation_matrix.

    Composition: Rz(-theta - PI/2) @ Rx(-phi) @ Rz(gamma)

    Parameters
    ----------
    phi
        Polar angle (rotation about the right axis).
    theta
        Azimuthal angle (rotation about the z axis).
    gamma
        Roll angle (rotation about the camera vector).

    Returns
    -------
    np.ndarray
        3x3 float64 rotation matrix.
    """
    PI = np.pi

    # Rz(-theta - PI/2)
    a1 = -theta - PI / 2
    c1, s1 = np.cos(a1), np.sin(a1)
    Rz1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]], dtype=np.float64)

    # Rx(-phi) — rotation about the x-axis
    c2, s2 = np.cos(-phi), np.sin(-phi)
    Rx = np.array([[1, 0, 0], [0, c2, -s2], [0, s2, c2]], dtype=np.float64)

    # Rz(gamma)
    c3, s3 = np.cos(gamma), np.sin(gamma)
    Rz2 = np.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]], dtype=np.float64)

    return Rz2 @ Rx @ Rz1
