"""NumPy / Metal buffer conversion helpers and VMobject geometry utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

try:
    import mapbox_earcut as _earcut
except ImportError:  # pragma: no cover - mapbox_earcut ships with manim
    _earcut = None

if TYPE_CHECKING:
    import numpy.typing as npt

# ---------------------------------------------------------------------------
# Pre-computed Bernstein basis coefficients for n_samples=64
# These are constant — computing them once avoids repeated power ops.
# ---------------------------------------------------------------------------
N_CURVE_SAMPLES = 64

# A fill is treated as planar (lying in a z = const plane, e.g. all 2D shapes
# and text glyphs) when its z extent is below this.  Planar fills use a robust
# ear-clipping triangulation; genuinely 3D geometry keeps the centroid fan.
_PLANAR_Z_EPS = 1e-6

_t = np.linspace(0, 1, N_CURVE_SAMPLES, dtype=np.float64).reshape(N_CURVE_SAMPLES, 1, 1)
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
    curve_pts = _B0 * ctrl[:, 0] + _B1 * ctrl[:, 1] + _B2 * ctrl[:, 2] + _B3 * ctrl[:, 3]

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


def _subpath_rings(points: npt.NDArray[np.float64]) -> list[npt.NDArray[np.float64]]:
    """Split a VMobject point array into linearized closed rings (one per subpath).

    A new subpath starts wherever consecutive cubic Bézier curves are not
    connected end-to-start.  Each ring is returned as a deduplicated (M, 2)
    polyline in the XY plane.
    """
    n_curves = len(points) // 4
    if n_curves == 0:
        return []
    ctrl = points[: n_curves * 4, :3].reshape(n_curves, 4, 3)

    # Subpath boundaries: curve k's end point differs from curve k+1's start.
    starts = [0]
    for k in range(n_curves - 1):
        if not np.allclose(ctrl[k, 3], ctrl[k + 1, 0], atol=1e-6):
            starts.append(k + 1)
    starts.append(n_curves)

    rings: list[npt.NDArray[np.float64]] = []
    for s in range(len(starts) - 1):
        seg = ctrl[starts[s] : starts[s + 1]].reshape(-1, 3)
        lin = _linearize_beziers_batch(seg)[:, :2]
        # Drop consecutive duplicates (incl. the closing point).
        keep = np.ones(len(lin), dtype=bool)
        keep[1:] = np.any(np.abs(np.diff(lin, axis=0)) > 1e-9, axis=1)
        lin = lin[keep]
        if len(lin) >= 3:
            rings.append(lin)
    return rings


def _point_in_polygon(p: npt.NDArray[np.float64], poly: npt.NDArray[np.float64]) -> bool:
    """Vectorized even-odd ray-cast point-in-polygon test (XY plane)."""
    x, y = float(p[0]), float(p[1])
    px, py = poly[:, 0], poly[:, 1]
    pxj, pyj = np.roll(px, 1), np.roll(py, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = (pxj - px) * (y - py) / (pyj - py) + px
    cond = ((py > y) != (pyj > y)) & (x < x_cross)
    return bool(np.count_nonzero(cond) % 2 == 1)


def _triangulate_planar_fill(
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float32] | None:
    """Ear-clip a planar (XY) VMobject fill, handling nested holes.

    Returns an (M, 3) float32 triangle-soup at the fill's constant z, or
    ``None`` if ear clipping is unavailable or fails (caller then falls back
    to the centroid fan).

    Why this exists: the centroid triangle-fan produces long sub-pixel slivers
    for small, thin-stroked shapes (e.g. superscript glyphs in ``MathTex``).
    GPU fixed-point rasterization drops those slivers — position-dependently —
    so such glyphs vanish.  Ear clipping yields well-shaped triangles that
    rasterize robustly, matching Cairo.
    """
    if _earcut is None:
        return None
    rings = _subpath_rings(points)
    if not rings:
        return None

    # Nesting depth via vertex containment: a ring at even depth is a filled
    # region, odd depth is a hole.  Using a vertex (not the centroid, which can
    # fall inside an annular hole) makes the test robust.
    r = len(rings)
    contains = np.zeros((r, r), dtype=bool)  # contains[j, i]: ring j holds ring i
    for i in range(r):
        v = rings[i][0]
        for j in range(r):
            if i != j and _point_in_polygon(v, rings[j]):
                contains[j, i] = True
    depth = contains.sum(axis=0)

    z_const = float(np.mean(points[:, 2])) if points.shape[1] > 2 else 0.0
    # 2D geometry projects z straight to NDC depth; a tiny *negative* z from
    # float noise (common in LaTeX-derived glyph points, ~1e-17) would be
    # clipped by Metal's near plane (z_ndc < 0), dropping the whole fill.
    # Snap near-zero z to exactly 0 so planar fills always survive.
    if abs(z_const) < _PLANAR_Z_EPS:
        z_const = 0.0
    tri_parts: list[npt.NDArray[np.float64]] = []
    for i in range(r):
        if depth[i] % 2 != 0:
            continue  # a hole — emitted together with its parent fill ring
        comp = [rings[i]]
        ends = [len(rings[i])]
        for k in range(r):
            if depth[k] == depth[i] + 1 and contains[i, k]:
                comp.append(rings[k])
                ends.append(ends[-1] + len(rings[k]))
        verts = np.concatenate(comp).astype(np.float64)
        try:
            idx = _earcut.triangulate_float64(verts, np.array(ends, dtype=np.uint32))
        except Exception:
            return None
        if len(idx):
            tri_parts.append(verts[np.asarray(idx, dtype=np.intp)].reshape(-1, 2))

    if not tri_parts:
        return None
    flat = np.concatenate(tri_parts)
    out = np.empty((len(flat), 3), dtype=np.float32)
    out[:, :2] = flat
    out[:, 2] = z_const
    return out


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

    # Add feather for stroke edge anti-aliasing (~1 pixel at default zoom).
    # Cap feather at the half-width so thin strokes (e.g. 0.5 on 3D surfaces)
    # aren't overwhelmed — keeps them proportionally thin like Cairo.
    expanded_hw = half_w + min(_STROKE_FEATHER, half_w)

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
# Normal computation utilities (vectorized, no Python loops over vertices)
# ---------------------------------------------------------------------------


def _normalize_vectors(v: npt.NDArray) -> npt.NDArray:
    """Normalize an (N, 3) array of vectors to unit length.

    Zero-length vectors are replaced with (0, 0, 1) — the default
    camera-facing normal for degenerate geometry.

    Returns float32 array.
    """
    norms = np.sqrt(np.sum(v * v, axis=1, keepdims=True))
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    result = (v / safe_norms).astype(np.float32)
    # Replace degenerate normals with camera-facing default
    degen = (norms < 1e-12).ravel()
    if np.any(degen):
        result[degen] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return result


def _cross_3d(a: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    """Vectorized cross product of (N, 3) arrays. Returns (N, 3) float64."""
    result = np.empty_like(a)
    result[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    result[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    result[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return result


def compute_face_normals(
    triangles: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute per-vertex face normals for flat-shaded triangle geometry.

    Each group of 3 consecutive vertices forms a triangle.  The face
    normal is the normalized cross product of two edges and is assigned
    to all 3 vertices (flat shading).

    Parameters
    ----------
    triangles
        (N*3, 3) float32 vertex positions, where every 3 rows form a triangle.

    Returns
    -------
    np.ndarray
        (N*3, 3) float32 unit normals, one per vertex.
    """
    n_verts = len(triangles)
    if n_verts == 0:
        return np.empty((0, 3), dtype=np.float32)

    n_tris = n_verts // 3
    if n_tris == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Reshape to (n_tris, 3_verts, 3_xyz)
    tris = triangles[: n_tris * 3].reshape(n_tris, 3, 3).astype(np.float64)
    v0 = tris[:, 0]
    v1 = tris[:, 1]
    v2 = tris[:, 2]

    # Edge vectors
    e1 = v1 - v0  # (n_tris, 3)
    e2 = v2 - v0  # (n_tris, 3)

    # Cross product e1 x e2 — gives un-normalized face normal
    raw_normals = _cross_3d(e1, e2)

    # Normalize to unit length
    face_normals = _normalize_vectors(raw_normals)  # (n_tris, 3) float32

    # Replicate each face normal to all 3 vertices of its triangle
    per_vertex = np.repeat(face_normals, 3, axis=0)  # (n_tris*3, 3)
    return per_vertex


def compute_smooth_fan_normals(
    triangles: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute smooth per-vertex normals for centroid-based triangle fan geometry.

    In a fan from ``_polyline_to_fan``, every triangle shares vertex 0
    (the centroid) and adjacent triangles share edge vertices.  This
    function averages face normals at shared vertices so the fragment
    shader interpolates smoothly, eliminating the flat-faceted look.

    Fan layout (N triangles)::

        tri i  →  [centroid,  polyline[i],  polyline[i+1 mod N]]

    Smooth normals:

    * **centroid**: mean of all face normals
    * **edge vertex p_i** (appears in tri *i* and tri *i−1*):
      mean of ``face_normal[i]`` and ``face_normal[i−1 mod N]``

    Parameters
    ----------
    triangles
        (N*3, 3) float32 vertex positions from ``_polyline_to_fan``.

    Returns
    -------
    np.ndarray
        (N*3, 3) float32 unit normals, smoothly interpolated at shared vertices.
    """
    n_verts = len(triangles)
    if n_verts < 3:
        return np.empty((0, 3), dtype=np.float32)

    n_tris = n_verts // 3
    if n_tris < 2:
        # Single triangle — just use face normal
        return compute_face_normals(triangles)

    tris = triangles[: n_tris * 3].reshape(n_tris, 3, 3).astype(np.float64)
    v0 = tris[:, 0]  # centroids
    v1 = tris[:, 1]  # current edge vertex (polyline[i])
    v2 = tris[:, 2]  # next edge vertex (polyline[i+1])

    e1 = v1 - v0
    e2 = v2 - v0
    raw = _cross_3d(e1, e2)
    face_normals = _normalize_vectors(raw).astype(np.float64)  # (n_tris, 3)

    # Centroid normal: mean of all face normals
    centroid_n = _normalize_vectors(face_normals.mean(axis=0, keepdims=True))

    # Edge vertex 1 (polyline[i]): average of face_normals[i] and face_normals[i-1]
    prev_fn = np.roll(face_normals, 1, axis=0)
    smooth_v1 = _normalize_vectors((face_normals + prev_fn) * 0.5)

    # Edge vertex 2 (polyline[i+1]): average of face_normals[i] and face_normals[i+1]
    next_fn = np.roll(face_normals, -1, axis=0)
    smooth_v2 = _normalize_vectors((face_normals + next_fn) * 0.5)

    # Assemble per-vertex normals
    result = np.empty((n_tris, 3, 3), dtype=np.float32)
    result[:, 0, :] = np.broadcast_to(centroid_n, (n_tris, 3))
    result[:, 1, :] = smooth_v1
    result[:, 2, :] = smooth_v2

    return result.reshape(-1, 3)


def compute_globally_smooth_normals(
    triangle_arrays: list[npt.NDArray[np.float32]],
) -> list[npt.NDArray[np.float32]]:
    """Smooth normals across sub-mobject boundaries by averaging at shared positions.

    Adjacent Surface patches share edge vertices at identical 3D positions but
    are tessellated independently.  ``compute_smooth_fan_normals`` only averages
    within a single fan, so normals at patch boundaries differ, causing visible
    embossing under Blinn-Phong lighting.

    This function gathers ALL triangle arrays, finds coincident vertices via
    spatial quantization (tolerance ~1e-4), accumulates face normals at shared
    positions, normalizes, and assigns the result back — eliminating boundary
    discontinuities.

    Parameters
    ----------
    triangle_arrays
        List of (M_i, 3) float32 vertex arrays — one per sub-mobject.
        Each array contains groups of 3 consecutive vertices forming triangles.

    Returns
    -------
    list[np.ndarray]
        Same-length list of (M_i, 3) float32 unit normals, one per input array.
        Coincident vertices across different arrays receive identical normals.
    """
    # Filter out empty arrays, tracking original indices
    non_empty: list[tuple[int, npt.NDArray[np.float32]]] = []
    vert_counts: list[int] = []
    for i, arr in enumerate(triangle_arrays):
        if arr is not None and len(arr) > 0:
            non_empty.append((i, arr))
            vert_counts.append(len(arr))

    # Prepare output list with empty arrays for empty inputs
    result: list[npt.NDArray[np.float32]] = [
        np.empty((0, 3), dtype=np.float32) if (a is None or len(a) == 0) else a
        for a in triangle_arrays
    ]

    if not non_empty:
        return [np.empty((0, 3), dtype=np.float32) for _ in triangle_arrays]

    # 1. Concatenate all non-empty arrays
    all_verts = np.concatenate([arr for _, arr in non_empty], axis=0)
    total_verts = len(all_verts)

    # 2. Compute per-face normals via cross product
    n_tris = total_verts // 3
    if n_tris == 0:
        return [
            np.empty((len(a) if a is not None else 0, 3), dtype=np.float32)
            for a in triangle_arrays
        ]

    tris = all_verts[: n_tris * 3].reshape(n_tris, 3, 3).astype(np.float64)
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    raw_face = _cross_3d(e1, e2)
    face_normals = _normalize_vectors(raw_face)  # (n_tris, 3) float32

    # 3. Replicate face normals to per-vertex
    per_vert_normals = np.repeat(face_normals, 3, axis=0).astype(np.float64)

    # 4. Quantize vertex positions for spatial matching (tolerance ~1e-4)
    verts_f64 = all_verts[: n_tris * 3].astype(np.float64)
    quantized = np.round(verts_f64 * 10000.0).astype(np.int64)

    # Scalar key for np.unique — combine x, y, z into one int64
    # Range: coordinates typically in [-100, 100] * 10000 = [-1e6, 1e6]
    # So x*1e14 + y*1e7 + z fits comfortably in int64
    keys = quantized[:, 0] * 10_000_000_000_000 + quantized[:, 1] * 10_000_000 + quantized[:, 2]

    _, inverse = np.unique(keys, return_inverse=True)

    # 5. Accumulate normals at shared positions
    n_unique = inverse.max() + 1
    accum = np.zeros((n_unique, 3), dtype=np.float64)
    np.add.at(accum, inverse, per_vert_normals)

    # 6. Normalize accumulated normals
    smooth = _normalize_vectors(accum)  # (n_unique, 3) float32

    # 7. Map back to per-vertex
    smooth_per_vert = smooth[inverse]  # (n_tris*3, 3) float32

    # 8. Split back to per-sub-mobject arrays
    split_points = np.cumsum(vert_counts[:-1])
    chunks = np.split(smooth_per_vert, split_points)

    for chunk_idx, (orig_idx, _) in enumerate(non_empty):
        result[orig_idx] = chunks[chunk_idx]

    return result


def compute_stroke_face_normals(
    stroke_quads: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute per-vertex face normals for stroke quad geometry.

    Stroke geometry is organized as 6 vertices per segment (2 triangles
    forming a quad: v0,v1,v2  v3,v4,v5).  Each segment gets a single
    face normal computed from the first triangle, replicated to all 6 vertices.

    Parameters
    ----------
    stroke_quads
        (N*6, 3) float32 vertex positions, where every 6 rows form a quad
        (two triangles).

    Returns
    -------
    np.ndarray
        (N*6, 3) float32 unit normals, one per vertex.
    """
    n_verts = len(stroke_quads)
    if n_verts == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Each quad segment = 6 vertices (2 triangles)
    n_segments = n_verts // 6
    if n_segments == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Reshape to (n_segments, 6, 3)
    segs = stroke_quads[: n_segments * 6].reshape(n_segments, 6, 3).astype(np.float64)

    # Use first triangle of each quad (vertices 0, 1, 2) for face normal
    v0 = segs[:, 0]
    v1 = segs[:, 1]
    v2 = segs[:, 2]

    e1 = v1 - v0
    e2 = v2 - v0
    raw_normals = _cross_3d(e1, e2)
    face_normals = _normalize_vectors(raw_normals)  # (n_segments, 3)

    # Replicate to all 6 vertices per segment
    per_vertex = np.repeat(face_normals, 6, axis=0)  # (n_segments*6, 3)
    return per_vertex


# ---------------------------------------------------------------------------
# Analytic normals for known primitive types
# ---------------------------------------------------------------------------


def compute_sphere_normals(
    vertices: npt.NDArray[np.float32],
    center: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float32]:
    """Compute analytic normals for sphere vertices.

    For a sphere, the outward normal at any point is simply the normalized
    vector from the center to that point: ``n = normalize(p - center)``.

    Parameters
    ----------
    vertices
        (N, 3) float32 vertex positions on or near the sphere surface.
    center
        (3,) sphere center. If None, estimated as the mean of all vertices.

    Returns
    -------
    np.ndarray
        (N, 3) float32 unit normals pointing outward from the sphere center.
    """
    if len(vertices) == 0:
        return np.empty((0, 3), dtype=np.float32)

    if center is None:
        center = np.mean(vertices, axis=0).astype(np.float64)
    else:
        center = np.asarray(center, dtype=np.float64)

    # Vector from center to each vertex
    diff = vertices.astype(np.float64) - center[np.newaxis, :]
    return _normalize_vectors(diff)


def compute_torus_normals(
    vertices: npt.NDArray[np.float32],
    major_radius: float,
    center: npt.NDArray[np.float64] | None = None,
    axis: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float32]:
    """Compute analytic normals for torus vertices.

    For a torus with major radius R centered at ``center`` with tube axis
    along ``axis``, the outward normal at point p is::

        n = normalize(p - nearest_point_on_ring)

    where ``nearest_point_on_ring`` is the closest point on the torus
    center ring to p.

    Parameters
    ----------
    vertices
        (N, 3) float32 vertex positions on or near the torus surface.
    major_radius
        The major radius R (distance from torus center to tube center ring).
    center
        (3,) torus center. Defaults to origin.
    axis
        (3,) unit vector along the torus symmetry axis. Defaults to (0, 0, 1).

    Returns
    -------
    np.ndarray
        (N, 3) float32 unit outward normals.
    """
    if len(vertices) == 0:
        return np.empty((0, 3), dtype=np.float32)

    if center is None:
        center = np.zeros(3, dtype=np.float64)
    else:
        center = np.asarray(center, dtype=np.float64)
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)

    pts = vertices.astype(np.float64) - center[np.newaxis, :]

    # Project each point onto the plane perpendicular to axis through origin
    # height = dot(pts, axis), projected = pts - height * axis
    heights = np.sum(pts * axis[np.newaxis, :], axis=1, keepdims=True)
    radial = pts - heights * axis[np.newaxis, :]

    # Normalize radial direction and scale to major radius for ring point
    radial_len = np.sqrt(np.sum(radial * radial, axis=1, keepdims=True))
    safe_radial_len = np.where(radial_len < 1e-12, 1.0, radial_len)
    radial_dir = radial / safe_radial_len

    # Nearest point on center ring
    ring_pts = radial_dir * major_radius

    # Normal = vector from ring point to surface point
    diff = pts - ring_pts
    return _normalize_vectors(diff)


def compute_cylinder_normals(
    vertices: npt.NDArray[np.float32],
    center: npt.NDArray[np.float64] | None = None,
    axis: npt.NDArray[np.float64] | None = None,
    height: float | None = None,
) -> npt.NDArray[np.float32]:
    """Compute analytic normals for cylinder vertices.

    Side vertices get radial normals perpendicular to the cylinder axis.
    Cap vertices (at the extremes of the height range along the axis) get
    normals pointing along +/- axis.

    If ``height`` is None, caps are not distinguished — all vertices get
    radial normals.

    Parameters
    ----------
    vertices
        (N, 3) float32 vertex positions.
    center
        (3,) cylinder center (midpoint). Defaults to origin.
    axis
        (3,) unit vector along cylinder axis. Defaults to (0, 0, 1).
    height
        Total height of the cylinder. If provided, vertices within a small
        tolerance of the top/bottom caps get axial normals.

    Returns
    -------
    np.ndarray
        (N, 3) float32 unit normals.
    """
    if len(vertices) == 0:
        return np.empty((0, 3), dtype=np.float32)

    if center is None:
        center = np.zeros(3, dtype=np.float64)
    else:
        center = np.asarray(center, dtype=np.float64)
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)

    pts = vertices.astype(np.float64) - center[np.newaxis, :]

    # Height along axis
    h = np.sum(pts * axis[np.newaxis, :], axis=1, keepdims=True)  # (N, 1)

    # Radial component (perpendicular to axis)
    radial = pts - h * axis[np.newaxis, :]
    normals = _normalize_vectors(radial)

    if height is not None:
        half_h = height / 2.0
        cap_tol = height * 0.01 + 1e-6  # 1% tolerance for cap detection

        h_flat = h.ravel()

        # Top cap: h ≈ +half_h
        top_mask = h_flat > (half_h - cap_tol)
        if np.any(top_mask):
            normals[top_mask] = axis.astype(np.float32)

        # Bottom cap: h ≈ -half_h
        bot_mask = h_flat < (-half_h + cap_tol)
        if np.any(bot_mask):
            normals[bot_mask] = (-axis).astype(np.float32)

    return normals


def compute_cone_normals(
    vertices: npt.NDArray[np.float32],
    apex: npt.NDArray[np.float64] | None = None,
    base_center: npt.NDArray[np.float64] | None = None,
    base_radius: float = 1.0,
    height: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Compute analytic normals for cone vertices.

    The cone normal is tilted outward by the half-angle from the axis.
    At the apex, normals point along the axis.

    Parameters
    ----------
    vertices
        (N, 3) float32 vertex positions.
    apex
        (3,) apex position. Defaults to (0, 0, height).
    base_center
        (3,) base center. Defaults to origin (0, 0, 0).
    base_radius
        Radius of the cone base.
    height
        Height from base center to apex.

    Returns
    -------
    np.ndarray
        (N, 3) float32 unit outward normals.
    """
    if len(vertices) == 0:
        return np.empty((0, 3), dtype=np.float32)

    if base_center is None:
        base_center = np.zeros(3, dtype=np.float64)
    else:
        base_center = np.asarray(base_center, dtype=np.float64)
    if apex is None:
        apex = base_center + np.array([0.0, 0.0, height], dtype=np.float64)
    else:
        apex = np.asarray(apex, dtype=np.float64)

    axis = apex - base_center
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-12:
        # Degenerate cone — return camera-facing normals
        out = np.empty((len(vertices), 3), dtype=np.float32)
        out[:] = [0.0, 0.0, 1.0]
        return out
    axis = axis / axis_len

    # Slant angle: tan(alpha) = base_radius / height
    # The normal is tilted by alpha from the radial direction
    slant_len = np.sqrt(base_radius**2 + height**2)
    cos_alpha = base_radius / slant_len  # axial component of normal
    sin_alpha = height / slant_len  # radial component of normal

    pts = vertices.astype(np.float64) - base_center[np.newaxis, :]

    # Height along axis from base
    h = np.sum(pts * axis[np.newaxis, :], axis=1, keepdims=True)

    # Radial component (perpendicular to axis)
    radial = pts - h * axis[np.newaxis, :]
    radial_len = np.sqrt(np.sum(radial * radial, axis=1, keepdims=True))
    safe_radial_len = np.where(radial_len < 1e-12, 1.0, radial_len)
    radial_dir = radial / safe_radial_len

    # Cone side normal: sin_alpha * radial_dir + cos_alpha * axis
    normals_f64 = sin_alpha * radial_dir + cos_alpha * axis[np.newaxis, :]
    normals = _normalize_vectors(normals_f64)

    # Base cap: vertices at h ≈ 0
    cap_tol = axis_len * 0.01 + 1e-6
    h_flat = h.ravel()
    base_mask = h_flat < cap_tol
    if np.any(base_mask):
        normals[base_mask] = (-axis).astype(np.float32)

    # Apex: vertices at h ≈ height (degenerate radial direction)
    apex_mask = h_flat > (axis_len - cap_tol)
    if np.any(apex_mask):
        normals[apex_mask] = axis.astype(np.float32)

    return normals


# ---------------------------------------------------------------------------
# Generic parametric surface normals via finite differences
# ---------------------------------------------------------------------------


def compute_surface_normals_finite_diff(
    func: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    u: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    eps: float = 1e-6,
) -> npt.NDArray[np.float32]:
    """Compute surface normals via finite differences for a parametric surface.

    For ``Surface(func)`` where ``func(u, v) -> (x, y, z)``, this computes
    ``normalize(cross(df/du, df/dv))`` using central differences.

    All operations are vectorized with numpy — no Python loops over vertices.

    Parameters
    ----------
    func
        Callable ``(u, v) -> np.ndarray`` of shape ``(N, 3)`` or broadcastable.
        Must accept arrays of u, v values and return corresponding (x, y, z)
        positions.  The function signature should be
        ``func(u: ndarray, v: ndarray) -> ndarray`` where the return shape
        is ``(N, 3)`` or ``(3,)`` if inputs are scalar.
    u
        (N,) array of u parameter values.
    v
        (N,) array of v parameter values.
    eps
        Small perturbation for finite difference computation.

    Returns
    -------
    np.ndarray
        (N, 3) float32 unit outward normals.
    """
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()

    if len(u) == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Central differences for partial derivatives
    # df/du ≈ (f(u+eps, v) - f(u-eps, v)) / (2*eps)
    f_u_plus = np.asarray(func(u + eps, v), dtype=np.float64).reshape(-1, 3)
    f_u_minus = np.asarray(func(u - eps, v), dtype=np.float64).reshape(-1, 3)
    df_du = (f_u_plus - f_u_minus) / (2.0 * eps)

    # df/dv ≈ (f(u, v+eps) - f(u, v-eps)) / (2*eps)
    f_v_plus = np.asarray(func(u, v + eps), dtype=np.float64).reshape(-1, 3)
    f_v_minus = np.asarray(func(u, v - eps), dtype=np.float64).reshape(-1, 3)
    df_dv = (f_v_plus - f_v_minus) / (2.0 * eps)

    # Normal = normalize(cross(df/du, df/dv))
    raw = _cross_3d(df_du, df_dv)
    return _normalize_vectors(raw)


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


def vmobject_to_triangles_with_normals(
    points: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Convert VMobject Bézier points to triangle fan vertices plus face normals.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(vertices, normals)`` — both (N, 3) float32.  Normals are flat-shaded
        face normals computed from cross products of triangle edges.
    """
    verts = vmobject_to_triangles(points)
    normals = compute_face_normals(verts)
    return verts, normals


def vmobject_to_stroke_quads_with_normals(
    points: npt.NDArray[np.float64],
    stroke_width: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Convert VMobject Bézier points to stroke quads plus face normals.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(vertices, normals)`` — both (N, 3) float32.
    """
    verts = vmobject_to_stroke_quads(points, stroke_width)
    normals = compute_stroke_face_normals(verts)
    return verts, normals


# ---------------------------------------------------------------------------
# Batch tessellation — process many objects in one NumPy call
# ---------------------------------------------------------------------------


def batch_tessellate(
    items: list[tuple[npt.NDArray[np.float64], float | None]],
    *,
    compute_normals: bool = False,
) -> list[tuple]:
    """Tessellate fill + stroke for many VMobjects in one fully-batched operation.

    All fan and stroke geometry is constructed in single vectorized NumPy
    operations across ALL objects simultaneously — no per-object Python loops.

    Parameters
    ----------
    items
        List of ``(points, stroke_width_or_None)`` per VMobject.
        *stroke_width* is the scene-unit stroke width (already scaled by 0.01),
        or ``None`` to skip stroke tessellation.
    compute_normals
        If True, per-vertex face normals are computed and returned alongside
        vertex data.  Default False for backward compatibility.

    Returns
    -------
    list[tuple]
        When ``compute_normals=False`` (default):
            ``(fill_triangles, stroke_quads_or_None)`` per input.
        When ``compute_normals=True``:
            ``(fill_triangles, fill_normals, stroke_quads_or_None,
              stroke_normals_or_None)`` per input.
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
        if compute_normals:
            return [
                (empty, empty, None if sw is None else empty, None if sw is None else empty)
                for _, sw in items
            ]
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
    centroids = np.add.reduceat(all_lin, nz_starts, axis=0) / nz_sizes[:, None]
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

    # Compute fan normals if requested — vectorized over ALL triangles at once
    all_fan_normals_flat = None
    if compute_normals:
        all_fan_normals_flat = _compute_fan_normals_batched(all_fan)

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
    stroke_results_per_nz: list[npt.NDArray[np.float32] | None] = [None] * len(nz_indices)
    stroke_normals_per_nz: list[npt.NDArray[np.float32] | None] = [None] * len(nz_indices)

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

            # Add feather for stroke edge anti-aliasing (capped for thin strokes)
            expanded_hw = hw + np.minimum(_STROKE_FEATHER, hw)

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

            # Compute stroke normals if requested
            all_stroke_normals_flat = None
            if compute_normals:
                all_stroke_normals_flat = _compute_stroke_normals_batched(quads)

            # Split stroke quads back per stroke object
            all_quads_flat = quads.reshape(-1, 3)
            seg_vert_counts = np.array(stroke_seg_counts)
            quad_splits = np.cumsum(seg_vert_counts[:-1] * 6)
            stroke_chunks = np.split(all_quads_flat, quad_splits)

            stroke_normal_chunks = None
            if compute_normals and all_stroke_normals_flat is not None:
                stroke_normal_chunks = np.split(all_stroke_normals_flat, quad_splits)

            chunk_idx = 0
            for k, j_nz in enumerate(stroke_nz_idx):
                if stroke_seg_counts[k] > 0:
                    stroke_results_per_nz[j_nz] = stroke_chunks[chunk_idx]
                    if stroke_normal_chunks is not None:
                        stroke_normals_per_nz[j_nz] = stroke_normal_chunks[chunk_idx]
                    chunk_idx += 1

    # --- Phase 5: Assemble results per input object ---
    # Split fan triangles per non-zero object
    fan_chunks = np.split(all_fan_flat, fan_splits)
    fan_normal_chunks = None
    if compute_normals and all_fan_normals_flat is not None:
        fan_normal_chunks = np.split(all_fan_normals_flat, fan_splits)

    results: list[tuple] = []
    nz_pos = 0  # position in nz_indices / fan_chunks / stroke_results_per_nz

    for i, (_, stroke_width) in enumerate(items):
        nc = curve_counts[i]
        if nc == 0:
            empty = np.empty((0, 3), dtype=np.float32)
            if compute_normals:
                results.append(
                    (
                        empty,
                        empty,
                        None if stroke_width is None else empty,
                        None if stroke_width is None else empty,
                    )
                )
            else:
                results.append((empty, None if stroke_width is None else empty))
            continue

        fill_tris = fan_chunks[nz_pos]
        # Planar fills (2D shapes, text glyphs) use robust ear clipping; the
        # centroid fan is kept for genuine 3D geometry and as a fallback.
        # The normal-computing path keeps the fan (normals are derived from it).
        if not compute_normals:
            pts_i = items[i][0]
            z = pts_i[:, 2]
            if float(z.max() - z.min()) <= _PLANAR_Z_EPS:
                earcut_tris = _triangulate_planar_fill(pts_i)
                if earcut_tris is not None and len(earcut_tris) > 0:
                    fill_tris = earcut_tris

        sq = stroke_results_per_nz[nz_pos]
        if stroke_width is not None and sq is None:
            sq = np.empty((0, 3), dtype=np.float32)

        if compute_normals:
            fill_norms = (
                fan_normal_chunks[nz_pos]
                if fan_normal_chunks is not None
                else np.empty((0, 3), dtype=np.float32)
            )
            stroke_norms = stroke_normals_per_nz[nz_pos]
            if stroke_width is not None and stroke_norms is None:
                stroke_norms = np.empty((0, 3), dtype=np.float32)
            results.append((fill_tris, fill_norms, sq, stroke_norms))
        else:
            results.append((fill_tris, sq))
        nz_pos += 1

    return results


def _compute_fan_normals_batched(
    all_fan: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute face normals for all fan triangles at once.

    Parameters
    ----------
    all_fan
        (N, 3, 3) array — N triangles, each with 3 vertices of 3 coords.

    Returns
    -------
    np.ndarray
        (N*3, 3) float32 unit normals, flat-shaded (same normal for all
        3 vertices in each triangle).
    """
    n_tris = len(all_fan)
    if n_tris == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Extract vertices: v0 = centroid, v1 = current, v2 = next
    v0 = all_fan[:, 0, :].astype(np.float64)
    v1 = all_fan[:, 1, :].astype(np.float64)
    v2 = all_fan[:, 2, :].astype(np.float64)

    e1 = v1 - v0  # (N, 3)
    e2 = v2 - v0  # (N, 3)

    raw = _cross_3d(e1, e2)
    face_normals = _normalize_vectors(raw)  # (N, 3) float32

    # Replicate to 3 vertices per triangle
    return np.repeat(face_normals, 3, axis=0)  # (N*3, 3)


def _compute_stroke_normals_batched(
    quads: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute face normals for all stroke quads at once.

    Parameters
    ----------
    quads
        (n_segments, 6, 3) array — each segment is 6 vertices (2 triangles).

    Returns
    -------
    np.ndarray
        (n_segments*6, 3) float32 unit normals, same normal for all 6 vertices
        in each segment (flat shading).
    """
    n_segments = len(quads)
    if n_segments == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Use first triangle (vertices 0, 1, 2) of each quad for the face normal
    v0 = quads[:, 0, :].astype(np.float64)
    v1 = quads[:, 1, :].astype(np.float64)
    v2 = quads[:, 2, :].astype(np.float64)

    e1 = v1 - v0
    e2 = v2 - v0
    raw = _cross_3d(e1, e2)
    face_normals = _normalize_vectors(raw)  # (n_segments, 3)

    # Replicate to 6 vertices per segment
    return np.repeat(face_normals, 6, axis=0)  # (n_segments*6, 3)


# ---------------------------------------------------------------------------
# Projection matrices and rotation utilities
# ---------------------------------------------------------------------------


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


def build_rotation_matrix(phi: float, theta: float, gamma: float) -> npt.NDArray[np.float64]:
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
