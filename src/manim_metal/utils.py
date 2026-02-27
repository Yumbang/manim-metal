"""NumPy / Metal buffer conversion helpers and VMobject geometry utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


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

    # Linearize each cubic Bézier into line segments
    n_curves = len(points) // 4
    t_values = np.linspace(0, 1, 8, dtype=np.float64)  # 8 samples per curve

    linearized = []
    for i in range(n_curves):
        p0 = points[4 * i][:2]
        p1 = points[4 * i + 1][:2]
        p2 = points[4 * i + 2][:2]
        p3 = points[4 * i + 3][:2]

        # De Casteljau evaluation for cubic Bézier
        t = t_values[:, np.newaxis]
        mt = 1.0 - t
        curve_pts = mt**3 * p0 + 3 * mt**2 * t * p1 + 3 * mt * t**2 * p2 + t**3 * p3
        linearized.append(curve_pts)

    if not linearized:
        return np.empty((0, 2), dtype=np.float32)

    polyline = np.concatenate(linearized, axis=0)

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

    # Linearize
    n_curves = len(points) // 4
    t_values = np.linspace(0, 1, 8, dtype=np.float64)

    linearized = []
    for i in range(n_curves):
        p0 = points[4 * i][:2]
        p1 = points[4 * i + 1][:2]
        p2 = points[4 * i + 2][:2]
        p3 = points[4 * i + 3][:2]

        t = t_values[:, np.newaxis]
        mt = 1.0 - t
        curve_pts = mt**3 * p0 + 3 * mt**2 * t * p1 + 3 * mt * t**2 * p2 + t**3 * p3
        linearized.append(curve_pts)

    if not linearized:
        return np.empty((0, 2), dtype=np.float32)

    polyline = np.concatenate(linearized, axis=0)

    # Remove consecutive duplicates
    mask = np.ones(len(polyline), dtype=bool)
    mask[1:] = np.any(np.abs(polyline[1:] - polyline[:-1]) > 1e-8, axis=1)
    polyline = polyline[mask]

    if len(polyline) < 2:
        return np.empty((0, 2), dtype=np.float32)

    half_w = stroke_width / 2.0
    n_segments = len(polyline) - 1
    # 6 vertices per segment (2 triangles)
    quads = np.empty((n_segments * 6, 2), dtype=np.float32)

    for i in range(n_segments):
        a = polyline[i]
        b = polyline[i + 1]
        d = b - a
        length = np.linalg.norm(d)
        if length < 1e-10:
            quads[i * 6 : (i + 1) * 6] = a.astype(np.float32)
            continue
        # Normal perpendicular to segment
        n = np.array([-d[1], d[0]]) / length * half_w

        # Four corners of the quad
        p0 = (a + n).astype(np.float32)
        p1 = (a - n).astype(np.float32)
        p2 = (b + n).astype(np.float32)
        p3 = (b - n).astype(np.float32)

        # Two triangles: (p0, p1, p2) and (p1, p3, p2)
        quads[i * 6 + 0] = p0
        quads[i * 6 + 1] = p1
        quads[i * 6 + 2] = p2
        quads[i * 6 + 3] = p1
        quads[i * 6 + 4] = p3
        quads[i * 6 + 5] = p2

    return quads


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
