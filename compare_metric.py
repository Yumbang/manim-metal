# ruff: noqa: I001, E402
"""Quantitative Cairo-vs-Metal pixel comparison harness.

This is the objective function for the "essentially same as Cairo" goal.
For each scene in a battery, it renders with BOTH the Cairo camera and the
Metal camera (identical mobjects, identical camera params) and computes
pixel-level divergence metrics:

  - MAE        mean absolute error per channel (0..255)
  - RMSE       root-mean-square error per channel
  - maxdiff    max absolute channel difference
  - bad%       fraction of pixels whose max channel diff exceeds a threshold
  - bad%(core) same, but excluding a 1px-dilated edge band (isolates interior
               fill/color divergence from pure anti-aliasing seams)

Usage:
    uv run python compare_metric.py                 # FXAA per camera default
    uv run python compare_metric.py --fxaa off       # force Metal FXAA off
    uv run python compare_metric.py --fxaa on        # force Metal FXAA on
    uv run python compare_metric.py --save-diffs     # write diff heatmaps
    uv run python compare_metric.py --only 2d_circle # run one scene
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

import manim_metal  # noqa: F401  (activates patches)

from manim import (
    BLUE,
    DEGREES,
    GREEN,
    ORANGE,
    PURPLE,
    RED,
    TEAL,
    WHITE,
    YELLOW,
    Annulus,
    Arc,
    Circle,
    Cone,
    Cylinder,
    Dot,
    Ellipse,
    Line,
    ManimColor,
    Polygon,
    RegularPolygon,
    RoundedRectangle,
    Sphere,
    Square,
    Star,
    Surface,
    Torus,
    Triangle,
    config,
)
from manim.camera.camera import Camera as CairoCamera
from manim.camera.three_d_camera import ThreeDCamera as CairoThreeDCamera
from manim.mobject.types.vectorized_mobject import VMobject

from manim_metal.metal_camera import MetalCamera

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "media_compare")
PW, PH = 1280, 720
FW = 14.22
BAD_THRESHOLD = 12  # channel-diff above this counts as a "bad" pixel


# ---------------------------------------------------------------------------
# Scene battery — each returns a list of VMobjects + camera params
# ---------------------------------------------------------------------------


def _fam(mob) -> list[VMobject]:
    return [m for m in mob.get_family() if isinstance(m, VMobject) and len(m.points) >= 4]


def _flat(*mobs) -> list[VMobject]:
    out: list[VMobject] = []
    for m in mobs:
        out.extend(_fam(m))
    return out


def scene_2d_circle():
    c = Circle(radius=1.5).set_fill(BLUE, opacity=0.7).set_stroke(WHITE, width=4.0)
    return _flat(c), {}


def scene_2d_circle_thin():
    c = Circle(radius=1.5).set_fill(BLUE, opacity=1.0).set_stroke(WHITE, width=1.0)
    return _flat(c), {}


def scene_2d_square():
    s = Square(side_length=2.0).set_fill(RED, opacity=0.6).set_stroke(WHITE, width=4.0)
    return _flat(s), {}


def scene_2d_square_fillonly():
    s = Square(side_length=3.0).set_fill(GREEN, opacity=1.0).set_stroke(width=0)
    return _flat(s), {}


def scene_2d_triangle():
    t = Triangle().scale(2.0).set_fill(YELLOW, opacity=0.8).set_stroke(WHITE, width=3.0)
    return _flat(t), {}


def scene_2d_polygon():
    p = RegularPolygon(n=7).scale(2.0).set_fill(PURPLE, opacity=0.7).set_stroke(ORANGE, width=5.0)
    return _flat(p), {}


def scene_2d_star():
    s = Star(n=6, outer_radius=2.0).set_fill(TEAL, opacity=0.9).set_stroke(WHITE, width=2.0)
    return _flat(s), {}


def scene_2d_ellipse():
    e = Ellipse(width=4.0, height=2.0).set_fill(ORANGE, opacity=0.6).set_stroke(BLUE, width=4.0)
    return _flat(e), {}


def scene_2d_annulus():
    a = Annulus(inner_radius=1.0, outer_radius=2.0).set_fill(GREEN, opacity=0.8)
    return _flat(a), {}


def scene_2d_rounded_rect():
    r = RoundedRectangle(corner_radius=0.5, width=4.0, height=2.5)
    r.set_fill(BLUE, opacity=0.7).set_stroke(WHITE, width=3.0)
    return _flat(r), {}


def scene_2d_line():
    line = Line(np.array([-3.0, -1.0, 0.0]), np.array([3.0, 1.0, 0.0]), stroke_width=6.0)
    line.set_stroke(YELLOW)
    return _flat(line), {}


def scene_2d_arc():
    a = Arc(radius=2.0, start_angle=0, angle=math.pi * 1.3).set_stroke(RED, width=5.0)
    return _flat(a), {}


def scene_2d_dot():
    d = Dot(point=np.array([0.0, 0.0, 0.0]), radius=0.5).set_fill(WHITE, opacity=1.0)
    return _flat(d), {}


def scene_2d_polygon_concave():
    pts = [
        np.array([-2.0, -2.0, 0.0]),
        np.array([2.0, -2.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 2.0, 0.0]),
        np.array([-2.0, 2.0, 0.0]),
    ]
    p = Polygon(*pts).set_fill(PURPLE, opacity=0.8).set_stroke(WHITE, width=3.0)
    return _flat(p), {}


def scene_2d_overlapping():
    c1 = Circle(radius=1.2).shift(np.array([-0.8, 0.0, 0.0]))
    c1.set_fill(BLUE, opacity=0.5).set_stroke(WHITE, width=3.0)
    c2 = Circle(radius=1.2).shift(np.array([0.8, 0.0, 0.0]))
    c2.set_fill(RED, opacity=0.5).set_stroke(WHITE, width=3.0)
    sq = Square(side_length=1.5).set_fill(GREEN, opacity=0.4).set_stroke(WHITE, width=3.0)
    return _flat(c1, c2, sq), {}


def scene_2d_many():
    mobs = []
    for i in range(12):
        ang = i * 2 * math.pi / 12
        c = Circle(radius=0.6).shift(np.array([2.5 * math.cos(ang), 2.5 * math.sin(ang), 0.0]))
        c.set_fill(ManimColor.from_hsv((i / 12, 0.8, 0.9)), opacity=0.7)
        c.set_stroke(WHITE, width=2.0)
        mobs.append(c)
    return _flat(*mobs), {}


def scene_text():
    from manim import Text

    t = Text("Hello Metal", color=WHITE).scale(1.5)
    return _flat(t), {}


def scene_text_colored():
    from manim import VGroup, Text

    a = Text("Cairo", color=BLUE).scale(1.2).shift(np.array([0, 1.0, 0]))
    b = Text("vs Metal", color=ORANGE).scale(1.2).shift(np.array([0, -1.0, 0]))
    return _flat(VGroup(a, b)), {}


def scene_mathtex():
    from manim import MathTex

    m = MathTex(r"e^{i\pi} + 1 = 0", color=WHITE).scale(2.5)
    return _flat(m), {}


def scene_3d_circle():
    c = Circle(radius=1.5).set_fill(BLUE, opacity=0.7).set_stroke(WHITE, width=4.0)
    return _flat(c), {"phi": 60, "theta": -45, "zoom": 1.0}


def scene_3d_square():
    s = Square(side_length=2.5).set_fill(RED, opacity=0.8).set_stroke(WHITE, width=3.0)
    return _flat(s), {"phi": 50, "theta": -30, "zoom": 1.0}


def scene_lit_sphere():
    sp = Sphere(radius=1.5).set_color(BLUE)
    return _flat(sp), {"phi": 60, "theta": -45, "zoom": 1.2}


def scene_lit_torus():
    t = Torus(major_radius=1.5, minor_radius=0.5).set_color(GREEN)
    return _flat(t), {"phi": 60, "theta": -45, "zoom": 1.2}


def scene_lit_cylinder():
    cy = Cylinder(radius=1.0, height=2.0).set_color(RED)
    return _flat(cy), {"phi": 60, "theta": -45, "zoom": 1.2}


def scene_lit_cone():
    co = Cone().set_color(ORANGE)
    return _flat(co), {"phi": 60, "theta": -45, "zoom": 1.2}


def scene_lit_surface():
    def f(u, v):
        return np.array(
            [np.cos(u) * (3 + np.cos(v)), np.sin(u) * (3 + np.cos(v)), np.sin(v)]
        )

    s = Surface(f, u_range=[0, 2 * np.pi], v_range=[0, 2 * np.pi], resolution=(24, 24))
    s.set_color(TEAL)
    return _flat(s), {"phi": 60, "theta": -45, "zoom": 0.6}


SCENES = {
    "2d_circle": scene_2d_circle,
    "2d_circle_thin": scene_2d_circle_thin,
    "2d_square": scene_2d_square,
    "2d_square_fillonly": scene_2d_square_fillonly,
    "2d_triangle": scene_2d_triangle,
    "2d_polygon": scene_2d_polygon,
    "2d_star": scene_2d_star,
    "2d_ellipse": scene_2d_ellipse,
    "2d_annulus": scene_2d_annulus,
    "2d_rounded_rect": scene_2d_rounded_rect,
    "2d_line": scene_2d_line,
    "2d_arc": scene_2d_arc,
    "2d_dot": scene_2d_dot,
    "2d_concave": scene_2d_polygon_concave,
    "2d_overlapping": scene_2d_overlapping,
    "2d_many": scene_2d_many,
    "text": scene_text,
    "text_colored": scene_text_colored,
    "mathtex": scene_mathtex,
    "3d_circle": scene_3d_circle,
    "3d_square": scene_3d_square,
    "lit_sphere": scene_lit_sphere,
    "lit_torus": scene_lit_torus,
    "lit_cylinder": scene_lit_cylinder,
    "lit_cone": scene_lit_cone,
    "lit_surface": scene_lit_surface,
}


# ---------------------------------------------------------------------------
# Camera construction
# ---------------------------------------------------------------------------


def _metal_cam(params, fxaa):
    cam = MetalCamera(pixel_width=PW, pixel_height=PH, frame_width=FW, fxaa=fxaa)
    if params:
        cam.set_phi(params.get("phi", 0) * DEGREES)
        cam.set_theta(params.get("theta", 0) * DEGREES)
        cam.set_zoom(params.get("zoom", 1.0))
    return cam


def _cairo_cam(params):
    if params:
        return CairoThreeDCamera(
            pixel_width=PW,
            pixel_height=PH,
            frame_width=FW,
            phi=params.get("phi", 0) * DEGREES,
            theta=params.get("theta", 0) * DEGREES,
            zoom=params.get("zoom", 1.0),
            focal_distance=20.0,
        )
    return CairoCamera(pixel_width=PW, pixel_height=PH, frame_width=FW)


def _render(cam, mobs):
    cam.reset()
    cam.capture_mobjects(mobs)
    return np.array(cam.get_image()).astype(np.int16)  # (H,W,4)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _edge_mask(a, b):
    """Boolean mask of pixels on or adjacent to an anti-aliasing seam.

    A pixel is "edge" if either image has a local color gradient there.
    Used to separate AA-seam divergence from interior divergence.
    """
    rgb_a = a[:, :, :3]
    grad = np.zeros(a.shape[:2], dtype=bool)
    for img in (rgb_a, b[:, :, :3]):
        gy = np.any(np.abs(np.diff(img, axis=0)) > 8, axis=2)
        gx = np.any(np.abs(np.diff(img, axis=1)) > 8, axis=2)
        grad[:-1, :] |= gy
        grad[1:, :] |= gy
        grad[:, :-1] |= gx
        grad[:, 1:] |= gx
    return grad


def compare(a, b):
    diff = np.abs(a[:, :, :3] - b[:, :, :3])  # ignore alpha for opaque bg
    mae = float(diff.mean())
    rmse = float(np.sqrt((diff.astype(np.float64) ** 2).mean()))
    maxdiff = int(diff.max())
    perpix = diff.max(axis=2)  # worst channel per pixel
    bad = perpix > BAD_THRESHOLD
    bad_frac = float(bad.mean())
    edges = _edge_mask(a, b)
    core_bad = bad & ~edges
    core_frac = float(core_bad.sum()) / float((~edges).sum() + 1)
    return {
        "mae": mae,
        "rmse": rmse,
        "maxdiff": maxdiff,
        "bad%": bad_frac * 100,
        "bad%core": core_frac * 100,
        "edge%": float(edges.mean()) * 100,
    }


def _save_diff(name, a, b, fxaa):
    from PIL import Image

    diff = np.abs(a[:, :, :3] - b[:, :, :3]).max(axis=2)
    heat = np.zeros((a.shape[0], a.shape[1], 3), dtype=np.uint8)
    heat[:, :, 0] = np.clip(diff * 4, 0, 255)  # red = magnitude
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = "fxaa" if fxaa else "nofxaa"
    # Side-by-side: cairo | metal | diff
    strip = np.concatenate(
        [a[:, :, :3].astype(np.uint8), b[:, :, :3].astype(np.uint8), heat], axis=1
    )
    Image.fromarray(strip).save(os.path.join(OUTPUT_DIR, f"{name}_{tag}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fxaa", choices=["on", "off", "default"], default="off")
    ap.add_argument("--save-diffs", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    config.renderer = "metal"
    config.pixel_width = PW
    config.pixel_height = PH
    config.frame_width = FW

    fxaa = {"on": True, "off": False, "default": True}[args.fxaa]

    names = [args.only] if args.only else list(SCENES)
    print(f"{'scene':<20} {'MAE':>7} {'RMSE':>7} {'maxd':>5} {'bad%':>7} {'core%':>7} {'edge%':>6}")
    print("-" * 70)

    agg = []
    for name in names:
        mobs, params = SCENES[name]()
        # NOTE: build fresh mobjects per camera to avoid state mutation issues
        mobs_c, params_c = SCENES[name]()
        m = _render(_metal_cam(params, fxaa), mobs)
        c = _render(_cairo_cam(params_c), mobs_c)
        stats = compare(c, m)
        agg.append((name, stats))
        print(
            f"{name:<20} {stats['mae']:>7.2f} {stats['rmse']:>7.2f} "
            f"{stats['maxdiff']:>5d} {stats['bad%']:>6.2f}% {stats['bad%core']:>6.2f}% "
            f"{stats['edge%']:>5.1f}%"
        )
        if args.save_diffs:
            _save_diff(name, c, m, fxaa)

    print("-" * 70)
    mean_mae = np.mean([s["mae"] for _, s in agg])
    mean_core = np.mean([s["bad%core"] for _, s in agg])
    mean_bad = np.mean([s["bad%"] for _, s in agg])
    print(f"{'MEAN':<20} {mean_mae:>7.2f} {'':>7} {'':>5} {mean_bad:>6.2f}% {mean_core:>6.2f}%")
    print(f"\nFXAA={'on' if fxaa else 'off'}  threshold={BAD_THRESHOLD}")


if __name__ == "__main__":
    main()
