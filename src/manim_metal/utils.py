"""NumPy / Metal buffer conversion helpers and VMobject geometry utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def _linearize_beziers_batch(
    points_2d: npt.NDArray[np.float64],
    n_samples: int = 8,
) -> npt.NDArray[np.float64]:
    """Linearize all cubic Bézier curves in a single batched NumPy operation.

    Parameters
    ----------
    points_2d
        (n_curves*4, 2) array of 2D control points (already xy-sliced).
    n_samples
        Number of sample points per curve.

    Returns
    -------
    np.ndarray
        (n_curves * n_samples, 2) array of linearized points.
    """
    n_curves = len(points_2d) // 4
    if n_curves == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Reshape to (n_curves, 4, 2) — P0, P1, P2, P3 per curve
    ctrl = points_2d[: n_curves * 4].reshape(n_curves, 4, 2)

    # t values: (n_samples, 1, 1) for broadcasting against (n_curves, 4, 2)
    t = np.linspace(0, 1, n_samples, dtype=np.float64).reshape(n_samples, 1, 1)
    mt = 1.0 - t

    # De Casteljau cubic evaluation — fully vectorized
    # Result shape: (n_samples, n_curves, 2)
    curve_pts = (
        mt**3 * ctrl[:, 0]
        + 3 * mt**2 * t * ctrl[:, 1]
        + 3 * mt * t**2 * ctrl[:, 2]
        + t**3 * ctrl[:, 3]
    )

    # Transpose to (n_curves, n_samples, 2), then flatten to (n_curves*n_samples, 2)
    return curve_pts.transpose(1, 0, 2).reshape(-1, 2)


def vmobject_to_triangles(
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float32]:
    """Convert VMobject cubic Bézier points to triangle fan vertices for stencil fill.

    Manim VMobjects store points as sequences of cubic Bézier curves,
    each defined by 4 points (anchor, handle1, handle2, anchor). We linearize
    each curve and create a triangle fan from the origin (centroid) to each
    successive pair of linearized points — this is used for stencil-then-cover
    fill rendering.

    Parameters
    ----------
    points
        (N, 3) array of Bézier control points from VMobject.points

    Returns
    -------
    np.ndarray
        (M, 2) array of float32 xy positions forming triangles (3 verts each).
    """
    if len(points) < 4:
        return np.empty((0, 2), dtype=np.float32)

    # Extract xy and linearize all curves at once
    n_curves = len(points) // 4
    points_2d = points[: n_curves * 4, :2]

    polyline = _linearize_beziers_batch(points_2d, n_samples=32)

    # Remove consecutive duplicates
    mask = np.ones(len(polyline), dtype=bool)
    mask[1:] = np.any(np.abs(polyline[1:] - polyline[:-1]) > 1e-8, axis=1)
    polyline = polyline[mask]

    if len(polyline) < 3:
        return np.empty((0, 2), dtype=np.float32)

    # Triangle fan from centroid
    centroid = polyline.mean(axis=0)
    n = len(polyline)
    triangles = np.empty((n, 3, 2), dtype=np.float32)
    triangles[:, 0, :] = centroid.astype(np.float32)
    triangles[:, 1, :] = polyline.astype(np.float32)
    triangles[:, 2, :] = np.roll(polyline, -1, axis=0).astype(np.float32)

    return triangles.reshape(-1, 2)


def vmobject_to_stroke_quads(
    points: npt.NDArray[np.float64],
    stroke_width: float,
) -> npt.NDArray[np.float32]:
    """Convert VMobject Bézier points to thick-line quad strip for stroke rendering.

    Each segment of the linearized polyline is expanded into a quad (2 triangles)
    perpendicular to the line direction, with half-width = stroke_width / 2.

    Parameters
    ----------
    points
        (N, 3) array of Bézier control points.
    stroke_width
        Stroke width in scene units.

    Returns
    -------
    np.ndarray
        (M, 2) array of float32 xy positions forming triangles (6 verts per segment).
    """
    if len(points) < 4 or stroke_width <= 0:
        return np.empty((0, 2), dtype=np.float32)

    # Extract xy and linearize all curves at once
    n_curves = len(points) // 4
    points_2d = points[: n_curves * 4, :2]

    polyline = _linearize_beziers_batch(points_2d, n_samples=32)

    # Remove consecutive duplicates
    mask = np.ones(len(polyline), dtype=bool)
    mask[1:] = np.any(np.abs(polyline[1:] - polyline[:-1]) > 1e-8, axis=1)
    polyline = polyline[mask]

    if len(polyline) < 2:
        return np.empty((0, 2), dtype=np.float32)

    half_w = stroke_width / 2.0

    # Vectorized stroke expansion
    a = polyline[:-1]  # (n_segments, 2) — start of each segment
    b = polyline[1:]   # (n_segments, 2) — end of each segment
    d = b - a          # direction vectors
    lengths = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)  # segment lengths

    # Handle degenerate segments (length ≈ 0)
    safe_lengths = np.where(lengths < 1e-10, 1.0, lengths)

    # Normal perpendicular to segment direction, scaled by half_w
    normals = np.empty_like(d)
    normals[:, 0] = -d[:, 1] / safe_lengths * half_w
    normals[:, 1] = d[:, 0] / safe_lengths * half_w

    # Four corners of each quad: (n_segments, 2) each
    p0 = a + normals
    p1 = a - normals
    p2 = b + normals
    p3 = b - normals

    # Build 6 vertices per segment: two triangles (p0,p1,p2) and (p1,p3,p2)
    n_segments = len(a)
    quads = np.empty((n_segments, 6, 2), dtype=np.float32)
    quads[:, 0] = p0
    quads[:, 1] = p1
    quads[:, 2] = p2
    quads[:, 3] = p1
    quads[:, 4] = p3
    quads[:, 5] = p2

    # Zero out degenerate segments
    degen_mask = lengths < 1e-10
    if np.any(degen_mask):
        quads[degen_mask] = a[degen_mask, np.newaxis, :].astype(np.float32)

    return quads.reshape(-1, 2)


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
