# ruff: noqa: I001
"""Unified verification and stress-testing tool for manim-metal.

Replaces the scattered benchmark.py, profile_frame.py, and test_lit_e2e.py
scripts with a single comprehensive runner that tests 2D basics, 3D basics,
lighting, stress scenarios, and profiling.

Usage:
    uv run python verify.py              # run all sections
    uv run python verify.py --section 2d # run only 2D basics
    uv run python verify.py --section 3d --section lighting
    uv run python verify.py --compare    # side-by-side Cairo vs Metal
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Activate manim-metal patches before any manim imports
import manim_metal  # noqa: F401

from manim import (
    BLUE,
    DEGREES,
    GREEN,
    ORANGE,
    RED,
    TEAL,
    WHITE,
    Circle,
    Cone,
    Cylinder,
    ManimColor,
    Rectangle,
    Sphere,
    Square,
    Surface,
    Torus,
    config,
)
from manim.camera.camera import Camera as CairoCamera
from manim.camera.three_d_camera import ThreeDCamera as CairoThreeDCamera
from manim.mobject.types.vectorized_mobject import VMobject

from manim_metal.metal_camera import MetalCamera, _needs_lighting
from manim_metal.utils import batch_tessellate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "media_verify")
PIXEL_WIDTH = 1920
PIXEL_HEIGHT = 1080
FRAME_WIDTH = 14.22
FRAME_HEIGHT = 8.0

# Thresholds for PASS/FAIL
THRESHOLD_SINGLE = 100  # min non-background pixels for single objects
THRESHOLD_MULTI = 500  # min non-background pixels for multi-object scenes

# Comparison image constants (scale with resolution)
_LABEL_HEIGHT = max(24, PIXEL_HEIGHT // 20)
_GAP_WIDTH = max(4, PIXEL_WIDTH // 200)

# Available section names
ALL_SECTIONS = ("2d", "3d", "lighting", "stress", "profiling")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_manim_config() -> None:
    """Configure manim globals for Metal headless rendering."""
    config.renderer = "metal"
    config.pixel_width = PIXEL_WIDTH
    config.pixel_height = PIXEL_HEIGHT
    config.frame_width = FRAME_WIDTH
    config.frame_height = FRAME_HEIGHT


def _make_camera_2d() -> MetalCamera:
    """Create a fresh 2D Metal camera (no rotation)."""
    return MetalCamera(
        pixel_width=PIXEL_WIDTH,
        pixel_height=PIXEL_HEIGHT,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
    )


def _make_camera_3d(
    phi: float = 60.0,
    theta: float = -45.0,
    zoom: float = 1.0,
) -> MetalCamera:
    """Create a fresh 3D Metal camera with lighting defaults."""
    cam = MetalCamera(
        pixel_width=PIXEL_WIDTH,
        pixel_height=PIXEL_HEIGHT,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
    )
    cam.set_phi(phi * DEGREES)
    cam.set_theta(theta * DEGREES)
    cam.set_zoom(zoom)
    return cam


def _make_cairo_camera_2d() -> CairoCamera:
    """Create a fresh 2D Cairo camera with matching dimensions."""
    return CairoCamera(
        pixel_width=PIXEL_WIDTH,
        pixel_height=PIXEL_HEIGHT,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
    )


def _make_cairo_camera_3d(
    phi: float = 60.0,
    theta: float = -45.0,
    zoom: float = 1.0,
) -> CairoThreeDCamera:
    """Create a fresh 3D Cairo camera with matching parameters."""
    return CairoThreeDCamera(
        pixel_width=PIXEL_WIDTH,
        pixel_height=PIXEL_HEIGHT,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        phi=phi * DEGREES,
        theta=theta * DEGREES,
        zoom=zoom,
        focal_distance=20.0,
    )


def _renderable_family(mob) -> list[VMobject]:
    """Extract all renderable VMobject sub-mobjects from a mobject hierarchy."""
    return [m for m in mob.get_family() if isinstance(m, VMobject) and len(m.points) >= 4]


def _render_and_save(
    cam: MetalCamera,
    mobjects: list[VMobject],
    filename: str,
) -> tuple[int, str]:
    """Render mobjects with Metal, save PNG, return (pixel_count, filepath)."""
    cam.reset()
    cam.capture_mobjects(mobjects)
    img = cam.get_image()
    filepath = os.path.join(OUTPUT_DIR, filename)
    img.save(filepath)

    # Count non-background pixels
    pixels = np.array(img)
    bg = cam.background
    diff = np.any(pixels != bg, axis=2)
    count = int(np.sum(diff))
    return count, filepath


def _render_cairo(
    cam: CairoCamera | CairoThreeDCamera,
    mobjects: list[VMobject],
    filename: str,
) -> str:
    """Render mobjects with Cairo camera, save PNG, return filepath."""
    cam.reset()
    cam.capture_mobjects(mobjects)
    img = cam.get_image()
    filepath = os.path.join(OUTPUT_DIR, filename)
    img.save(filepath)
    return filepath


def _stitch_comparison(
    cairo_path: str,
    metal_path: str,
    output_name: str,
) -> str:
    """Create a labeled side-by-side comparison image. Returns output filepath."""
    cairo_img = Image.open(cairo_path)
    metal_img = Image.open(metal_path)

    w = cairo_img.width
    h = cairo_img.height
    total_w = w * 2 + _GAP_WIDTH
    total_h = h + _LABEL_HEIGHT

    comp = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 255))
    draw = ImageDraw.Draw(comp)

    # Use a scaled font size (fall back to default if truetype unavailable)
    font_size = max(16, h // 30)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Center labels above each panel
    for label, x_offset in [("Cairo", 0), ("Metal", w + _GAP_WIDTH)]:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        y_pad = (_LABEL_HEIGHT - text_h) // 2
        draw.text((x_offset + (w - text_w) // 2, y_pad), label, fill="white", font=font)

    # Paste images below labels
    comp.paste(cairo_img, (0, _LABEL_HEIGHT))
    comp.paste(metal_img, (w + _GAP_WIDTH, _LABEL_HEIGHT))

    filepath = os.path.join(OUTPUT_DIR, output_name)
    comp.save(filepath)
    return filepath


def _maybe_compare(
    cairo_cam: CairoCamera | CairoThreeDCamera | None,
    mobjects: list[VMobject],
    metal_path: str,
    name: str,
    saved: list[str],
) -> None:
    """If cairo_cam is set, render Cairo image and stitch side-by-side comparison."""
    if cairo_cam is None:
        return
    cairo_path = _render_cairo(cairo_cam, mobjects, f"cairo_{name}.png")
    comp_path = _stitch_comparison(cairo_path, metal_path, f"compare_{name}.png")
    saved.append(comp_path)


def _format_result(label: str, pixel_count: int, threshold: int) -> tuple[bool, str]:
    """Format a single test result line. Returns (passed, formatted_string)."""
    passed = pixel_count > threshold
    status = "PASS" if passed else "FAIL"
    return passed, f"  {label:<25s} {pixel_count:>8,} pixels  {status}"


# ---------------------------------------------------------------------------
# Section 1: 2D Basics
# ---------------------------------------------------------------------------


def run_2d_basics(compare: bool = False) -> tuple[bool, str]:
    """Test basic 2D rendering: Circle, Square, overlapping shapes."""
    lines: list[str] = []
    saved: list[str] = []
    all_passed = True

    cam = _make_camera_2d()
    cairo_cam = _make_cairo_camera_2d() if compare else None

    # -- Circle --
    circle = Circle(radius=1.5)
    circle.set_fill(BLUE, opacity=0.7)
    circle.set_stroke(WHITE, width=2.0)
    family = _renderable_family(circle)
    count, path = _render_and_save(cam, family, "2d_circle.png")
    passed, line = _format_result("Circle:", count, THRESHOLD_SINGLE)
    lines.append(line)
    saved.append(path)
    all_passed = all_passed and passed
    _maybe_compare(cairo_cam, family, path, "2d_circle", saved)

    # -- Square --
    square = Square(side_length=2.0)
    square.set_fill(RED, opacity=0.6)
    square.set_stroke(WHITE, width=2.0)
    family = _renderable_family(square)
    count, path = _render_and_save(cam, family, "2d_square.png")
    passed, line = _format_result("Square:", count, THRESHOLD_SINGLE)
    lines.append(line)
    saved.append(path)
    all_passed = all_passed and passed
    _maybe_compare(cairo_cam, family, path, "2d_square", saved)

    # -- Overlapping shapes --
    c1 = Circle(radius=1.0).shift(np.array([-0.8, 0.0, 0.0]))
    c1.set_fill(BLUE, opacity=0.5)
    c1.set_stroke(WHITE, width=1.5)

    c2 = Circle(radius=1.0).shift(np.array([0.8, 0.0, 0.0]))
    c2.set_fill(RED, opacity=0.5)
    c2.set_stroke(WHITE, width=1.5)

    sq = Square(side_length=1.5)
    sq.set_fill(GREEN, opacity=0.4)
    sq.set_stroke(WHITE, width=1.5)

    family = _renderable_family(c1) + _renderable_family(c2) + _renderable_family(sq)
    count, path = _render_and_save(cam, family, "2d_overlapping.png")
    passed, line = _format_result("Overlapping:", count, THRESHOLD_MULTI)
    lines.append(line)
    saved.append(path)
    all_passed = all_passed and passed
    _maybe_compare(cairo_cam, family, path, "2d_overlapping", saved)

    saved_str = ", ".join(os.path.basename(p) for p in saved)
    lines.append(f"  Saved: {saved_str}")
    return all_passed, "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 2: 3D Basics
# ---------------------------------------------------------------------------


def run_3d_basics(compare: bool = False) -> tuple[bool, str]:
    """Test basic 3D rendering: rotated circle, depth ordering."""
    lines: list[str] = []
    saved: list[str] = []
    all_passed = True

    # -- 3D Circle with camera rotation --
    cam = _make_camera_3d(phi=60, theta=-45, zoom=1.0)
    cairo_cam = _make_cairo_camera_3d(phi=60, theta=-45, zoom=1.0) if compare else None

    circle_3d = Circle(radius=1.5)
    circle_3d.set_fill(BLUE, opacity=0.7)
    circle_3d.set_stroke(WHITE, width=2.0)
    family = _renderable_family(circle_3d)
    count, path = _render_and_save(cam, family, "3d_circle.png")
    passed, line = _format_result("3D Circle:", count, THRESHOLD_SINGLE)
    lines.append(line)
    saved.append(path)
    all_passed = all_passed and passed
    _maybe_compare(cairo_cam, family, path, "3d_circle", saved)

    # -- Depth ordering: overlapping rectangles at different z --
    cam = _make_camera_3d(phi=30, theta=-30, zoom=1.0)
    cairo_cam = _make_cairo_camera_3d(phi=30, theta=-30, zoom=1.0) if compare else None

    r1 = Rectangle(width=3.0, height=2.0)
    r1.set_fill(RED, opacity=0.8)
    r1.set_stroke(WHITE, width=1.0)
    r1.shift(np.array([0.0, 0.0, -1.0]))

    r2 = Rectangle(width=3.0, height=2.0)
    r2.set_fill(GREEN, opacity=0.8)
    r2.set_stroke(WHITE, width=1.0)
    r2.shift(np.array([0.5, 0.3, 0.0]))

    r3 = Rectangle(width=3.0, height=2.0)
    r3.set_fill(BLUE, opacity=0.8)
    r3.set_stroke(WHITE, width=1.0)
    r3.shift(np.array([1.0, 0.6, 1.0]))

    family = _renderable_family(r1) + _renderable_family(r2) + _renderable_family(r3)
    count, path = _render_and_save(cam, family, "3d_depth_ordering.png")
    passed, line = _format_result("Depth ordering:", count, THRESHOLD_MULTI)
    lines.append(line)
    saved.append(path)
    all_passed = all_passed and passed
    _maybe_compare(cairo_cam, family, path, "3d_depth_ordering", saved)

    saved_str = ", ".join(os.path.basename(p) for p in saved)
    lines.append(f"  Saved: {saved_str}")
    return all_passed, "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 3: Lighting (Blinn-Phong)
# ---------------------------------------------------------------------------


def _run_lit_primitive(
    cam: MetalCamera,
    mob,
    label: str,
    filename: str,
) -> tuple[bool, str, str]:
    """Render a single lit primitive. Returns (passed, result_line, filepath)."""
    family = _renderable_family(mob)
    lit_count = sum(1 for m in family if _needs_lighting(m))
    count, path = _render_and_save(cam, family, filename)
    passed, line = _format_result(f"{label} (lit={lit_count}):", count, THRESHOLD_SINGLE)
    return passed, line, path


def run_lighting(compare: bool = False) -> tuple[bool, str]:
    """Test Blinn-Phong lighting on all 3D primitives."""
    lines: list[str] = []
    saved: list[str] = []
    all_passed = True

    cam = _make_camera_3d(phi=60, theta=-45, zoom=1.2)
    cairo_cam = _make_cairo_camera_3d(phi=60, theta=-45, zoom=1.2) if compare else None

    # -- Sphere --
    sphere = Sphere(radius=1.5)
    sphere.set_color(BLUE)
    passed, line, path = _run_lit_primitive(cam, sphere, "Sphere", "lit_sphere.png")
    lines.append(line)
    saved.append(path)
    all_passed = all_passed and passed
    _maybe_compare(cairo_cam, _renderable_family(sphere), path, "lit_sphere", saved)

    # -- Torus --
    torus = Torus(major_radius=1.5, minor_radius=0.5)
    torus.set_color(GREEN)
    passed, line, path = _run_lit_primitive(cam, torus, "Torus", "lit_torus.png")
    lines.append(line)
    saved.append(path)
    all_passed = all_passed and passed
    _maybe_compare(cairo_cam, _renderable_family(torus), path, "lit_torus", saved)

    # -- Cylinder --
    cylinder = Cylinder(radius=1.0, height=2.0)
    cylinder.set_color(RED)
    passed, line, path = _run_lit_primitive(cam, cylinder, "Cylinder", "lit_cylinder.png")
    lines.append(line)
    saved.append(path)
    all_passed = all_passed and passed
    _maybe_compare(cairo_cam, _renderable_family(cylinder), path, "lit_cylinder", saved)

    # -- Cone --
    cone = Cone()
    cone.set_color(ORANGE)
    passed, line, path = _run_lit_primitive(cam, cone, "Cone", "lit_cone.png")
    lines.append(line)
    saved.append(path)
    all_passed = all_passed and passed
    _maybe_compare(cairo_cam, _renderable_family(cone), path, "lit_cone", saved)

    # -- Parametric Surface --
    def param_func(u, v):
        x = np.cos(u) * (3 + np.cos(v))
        y = np.sin(u) * (3 + np.cos(v))
        z = np.sin(v)
        return np.array([x, y, z])

    surface = Surface(
        param_func,
        u_range=[0, 2 * np.pi],
        v_range=[0, 2 * np.pi],
        resolution=(24, 24),
    )
    surface.set_color(TEAL)
    # Zoom out a bit for the larger surface
    cam.set_zoom(0.6)
    if cairo_cam is not None:
        cairo_cam.set_zoom(0.6)
    passed, line, path = _run_lit_primitive(cam, surface, "Surface", "lit_surface.png")
    lines.append(line)
    saved.append(path)
    all_passed = all_passed and passed
    _maybe_compare(cairo_cam, _renderable_family(surface), path, "lit_surface", saved)

    saved_str = ", ".join(os.path.basename(p) for p in saved)
    lines.append(f"  Saved: {saved_str}")
    return all_passed, "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 4: Stress Test
# ---------------------------------------------------------------------------


def run_stress_test(compare: bool = False) -> tuple[bool, str]:
    """Render 50+ mixed lit/unlit objects and verify no crash."""
    lines: list[str] = []
    saved: list[str] = []
    all_passed = True

    cam = _make_camera_3d(phi=45, theta=-30, zoom=0.8)
    cairo_cam = _make_cairo_camera_3d(phi=45, theta=-30, zoom=0.8) if compare else None

    all_mobs: list[VMobject] = []

    # 20 2D circles at various positions (unlit)
    for i in range(20):
        angle = i * 2 * math.pi / 20
        r = 2.5
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        c = Circle(radius=0.3 + 0.02 * i)
        c.set_fill(
            color=ManimColor.from_hsv((i / 20, 0.8, 0.9)),
            opacity=0.5,
        )
        c.set_stroke(WHITE, width=1.0)
        c.shift(np.array([x, y, 0.0]))
        all_mobs.extend(_renderable_family(c))

    # 15 squares at different z-levels (unlit)
    for i in range(15):
        sq = Square(side_length=0.5)
        sq.set_fill(
            color=ManimColor.from_hsv((i / 15, 0.6, 0.8)),
            opacity=0.4,
        )
        sq.set_stroke(WHITE, width=0.8)
        x = -4.0 + (i % 5) * 2.0
        y = -2.0 + (i // 5) * 2.0
        z = -1.0 + i * 0.15
        sq.shift(np.array([x, y, z]))
        all_mobs.extend(_renderable_family(sq))

    # 10 small lit Spheres scattered around
    for i in range(10):
        angle = i * 2 * math.pi / 10
        sp = Sphere(radius=0.4)
        sp.set_color(ManimColor.from_hsv((i / 10, 0.7, 0.95)))
        x = 4.0 * math.cos(angle)
        y = 4.0 * math.sin(angle)
        z = math.sin(angle) * 1.5
        sp.shift(np.array([x, y, z]))
        all_mobs.extend(_renderable_family(sp))

    # 5 small Tori
    for i in range(5):
        t = Torus(major_radius=0.5, minor_radius=0.15)
        t.set_color(ManimColor.from_hsv(((i * 0.2 + 0.1) % 1.0, 0.8, 0.9)))
        x = -3.0 + i * 1.5
        t.shift(np.array([x, 3.0, 0.5 * i]))
        all_mobs.extend(_renderable_family(t))

    total_vmobs = len(all_mobs)
    count, path = _render_and_save(cam, all_mobs, "stress_mixed.png")
    passed, line = _format_result(f"{total_vmobs} mixed objects:", count, THRESHOLD_MULTI)
    lines.append(line)
    saved.append(path)
    all_passed = all_passed and passed
    _maybe_compare(cairo_cam, all_mobs, path, "stress_mixed", saved)

    saved_str = ", ".join(os.path.basename(p) for p in saved)
    lines.append(f"  Saved: {saved_str}")
    return all_passed, "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 5: Profiling
# ---------------------------------------------------------------------------


def _build_2d_objects(n: int) -> list[VMobject]:
    """Build n 2D objects (circles + squares) for profiling."""
    mobs: list[VMobject] = []
    for i in range(n):
        if i % 2 == 0:
            c = Circle(radius=0.15 + 0.002 * i)
            c.set_fill(
                color=ManimColor.from_hsv((i / n, 0.8, 0.9)),
                opacity=0.5,
            )
            c.set_stroke(WHITE, width=1.0)
        else:
            c = Square(side_length=0.3)
            c.set_fill(BLUE, opacity=0.3)
            c.set_stroke(WHITE, width=0.8)
        angle = i * 0.3
        r = 0.04 * i
        c.shift(np.array([r * math.cos(angle), r * math.sin(angle), 0.0]))
        mobs.extend(_renderable_family(c))
    return mobs


def _build_3d_lit_objects(n: int) -> list[VMobject]:
    """Build n lit Sphere objects for profiling."""
    mobs: list[VMobject] = []
    for i in range(n):
        sp = Sphere(radius=0.4)
        sp.set_color(ManimColor.from_hsv((i / n, 0.7, 0.9)))
        angle = i * 2 * math.pi / n
        sp.shift(np.array([3.0 * math.cos(angle), 3.0 * math.sin(angle), 0.0]))
        mobs.extend(_renderable_family(sp))
    return mobs


def run_profiling(compare: bool = False) -> tuple[bool, str]:
    """Profile frame timing: tessellation and full-frame for unlit and lit scenes."""
    lines: list[str] = []
    n_warmup = 3
    n_iters = 10

    # --- Unlit profiling (200 2D objects) ---
    n_2d = 200
    cam_2d = _make_camera_2d()
    vmobs_2d = _build_2d_objects(n_2d)

    # Warmup
    for _ in range(n_warmup):
        cam_2d.reset()
        cam_2d.capture_mobjects(vmobs_2d)

    # Tessellation timing
    tess_times: list[float] = []
    for _ in range(n_iters):
        cam_2d._geo_cache.clear()
        items = []
        for vmob in vmobs_2d:
            sw = vmob.get_stroke_width()
            scene_sw = sw * 0.01 if sw > 0 else None
            items.append((vmob.points, scene_sw))
        t0 = time.perf_counter()
        batch_tessellate(items)
        tess_times.append(time.perf_counter() - t0)
    tess_avg = np.mean(tess_times) * 1000

    # Full frame timing
    frame_times: list[float] = []
    for _ in range(n_iters):
        cam_2d._geo_cache.clear()
        cam_2d.reset()
        t0 = time.perf_counter()
        cam_2d.capture_mobjects(vmobs_2d)
        frame_times.append(time.perf_counter() - t0)
    frame_avg = np.mean(frame_times) * 1000

    lines.append(f"  Unlit ({len(vmobs_2d)} 2D vmobs from {n_2d} objects):")
    lines.append(f"    Tessellation:    {tess_avg:>7.2f} ms")
    lines.append(f"    Full frame:      {frame_avg:>7.2f} ms")

    # --- Lit profiling (20 Spheres) ---
    n_3d = 20
    cam_3d = _make_camera_3d(phi=60, theta=-45, zoom=0.7)
    vmobs_3d = _build_3d_lit_objects(n_3d)

    # Warmup
    for _ in range(n_warmup):
        cam_3d.reset()
        cam_3d.capture_mobjects(vmobs_3d)

    # Tessellation timing (with normals for lit objects)
    tess_times_3d: list[float] = []
    for _ in range(n_iters):
        cam_3d._geo_cache.clear()
        items = []
        for vmob in vmobs_3d:
            sw = vmob.get_stroke_width()
            scene_sw = sw * 0.01 if sw > 0 else None
            items.append((vmob.points, scene_sw))
        t0 = time.perf_counter()
        batch_tessellate(items, compute_normals=True)
        tess_times_3d.append(time.perf_counter() - t0)
    tess_avg_3d = np.mean(tess_times_3d) * 1000

    # Full frame timing
    frame_times_3d: list[float] = []
    for _ in range(n_iters):
        cam_3d._geo_cache.clear()
        cam_3d.reset()
        t0 = time.perf_counter()
        cam_3d.capture_mobjects(vmobs_3d)
        frame_times_3d.append(time.perf_counter() - t0)
    frame_avg_3d = np.mean(frame_times_3d) * 1000

    lines.append(f"  Lit ({len(vmobs_3d)} 3D vmobs from {n_3d} Spheres):")
    lines.append(f"    Tessellation:    {tess_avg_3d:>7.2f} ms")
    lines.append(f"    Full frame:      {frame_avg_3d:>7.2f} ms")

    # Profiling always "passes" as long as it runs without error
    return True, "\n".join(lines)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

SECTION_REGISTRY: dict[str, tuple[str, callable]] = {
    "2d": ("2D Basics", run_2d_basics),
    "3d": ("3D Basics", run_3d_basics),
    "lighting": ("Lighting", run_lighting),
    "stress": ("Stress Test", run_stress_test),
    "profiling": ("Profiling", run_profiling),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified verification and stress-testing tool for manim-metal.",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=list(ALL_SECTIONS),
        help="Run specific section(s). Can be repeated. Default: all.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Render each scene with both Cairo and Metal, produce side-by-side PNGs.",
    )
    args = parser.parse_args()

    sections = args.section if args.section else list(ALL_SECTIONS)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Configure manim
    _init_manim_config()

    print("=== manim-metal verify ===")
    if args.compare:
        print("  (compare mode: Cairo vs Metal side-by-side)")
    print()

    total_tests = 0
    total_passed = 0

    for section_key in sections:
        title, func = SECTION_REGISTRY[section_key]
        print(f"--- {title} ---")

        try:
            passed, details = func(compare=args.compare)
            print(details)
            # Count tests per section (each non-profiling section has tests)
            if section_key != "profiling":
                # Count result lines (lines with PASS or FAIL)
                result_lines = [ln for ln in details.split("\n") if "PASS" in ln or "FAIL" in ln]
                n_tests = len(result_lines)
                n_passed = sum(1 for ln in result_lines if "PASS" in ln)
                total_tests += n_tests
                total_passed += n_passed
            else:
                # Profiling counts as 1 test (did it run?)
                total_tests += 1
                total_passed += 1 if passed else 0
        except Exception:
            print(f"  ERROR: section '{section_key}' raised an exception:")
            traceback.print_exc()
            total_tests += 1
            # Count as failed

        print()

    print(f"=== Results: {total_passed}/{total_tests} PASS ===")
    sys.exit(0 if total_passed == total_tests else 1)


if __name__ == "__main__":
    main()
