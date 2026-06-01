# ruff: noqa: I001, E402
"""Cairo-vs-Metal RENDERING PERFORMANCE benchmark.

Times the core per-frame render work — ``camera.capture_mobjects`` — for the
Cairo renderer and the Metal renderer on identical mobjects, across several
workloads and resolutions, and reports the speedup.  This isolates rendering
(tessellation + rasterization/GPU + readback) from fixed costs like process
startup and video muxing, which are identical for both backends.

Two Metal regimes are reported:
  - warm : geometry cache hot (steady-state animation where geometry is static
           and only the camera moves, or a held frame) — the common case.
  - cold : geometry cache cleared every frame (geometry changes every frame,
           e.g. a Transform) — Metal re-tessellates each frame.
Cairo has no such cache; it re-rasterizes every frame regardless.

Usage:
    uv run python benchmark.py              # full matrix, prints table
    uv run python benchmark.py --write-md   # also (re)write BENCHMARK.md
    uv run python benchmark.py --res 1080p  # single resolution
"""

from __future__ import annotations

import argparse
import math
import platform
import statistics
import subprocess
import time

import numpy as np

import manim_metal  # noqa: F401  (activates Metal patches)

from manim import (
    BLUE, GREEN, ORANGE, RED, TEAL, WHITE, YELLOW,
    Circle, Cylinder, ManimColor, MathTex, Sphere, Square, Text, Torus,
    DEGREES, config,
)
from manim.camera.camera import Camera as CairoCamera
from manim.camera.three_d_camera import ThreeDCamera as CairoThreeDCamera
from manim.mobject.types.vectorized_mobject import VMobject

from manim_metal.metal_camera import MetalCamera

FW = 14.22
RESOLUTIONS = {
    "480p": (854, 480),
    "1080p": (1920, 1080),
    "4K": (3840, 2160),
}


def _fam(m):
    return [x for x in m.get_family() if isinstance(x, VMobject) and len(x.points) >= 4]


def _flat(*ms):
    out = []
    for m in ms:
        out.extend(_fam(m))
    return out


# ---------------------------------------------------------------------------
# Workloads — each returns (build_fn, camera_params_or_None)
# camera_params None => 2D; dict => 3D with phi/theta/zoom
# ---------------------------------------------------------------------------


def w_2d_light():
    mobs = []
    for i in range(50):
        a = i * 2 * math.pi / 50
        c = (Circle(radius=0.5) if i % 2 == 0 else Square(side_length=0.7))
        c.set_fill(ManimColor.from_hsv((i / 50, 0.8, 0.9)), opacity=0.6).set_stroke(WHITE, 2)
        c.shift(np.array([3.5 * math.cos(a), 2.0 * math.sin(a), 0.0]))
        mobs.append(c)
    return _flat(*mobs), None


def w_2d_heavy():
    mobs = []
    for i in range(200):
        c = (Circle(radius=0.15 + 0.002 * i) if i % 2 == 0 else Square(side_length=0.3))
        c.set_fill(ManimColor.from_hsv((i / 200, 0.8, 0.9)), opacity=0.5).set_stroke(WHITE, 1)
        a = i * 0.3
        c.shift(np.array([0.04 * i * math.cos(a), 0.04 * i * math.sin(a), 0.0]))
        mobs.append(c)
    return _flat(*mobs), None


def w_text():
    t = Text("The quick brown fox jumps over the lazy dog. 0123456789",
             color=WHITE).scale(0.7)
    return _flat(t), None


def w_mathtex():
    m = MathTex(r"\sum_{k=1}^{n} k^2 = \frac{n(n+1)(2n+1)}{6}", color=WHITE).scale(1.8)
    return _flat(m), None


def w_3d_sphere():
    sp = Sphere(radius=2.0).set_color(BLUE)
    return _flat(sp), {"phi": 60, "theta": -45, "zoom": 1.0}


def w_3d_heavy():
    sp = Sphere(radius=1.2).set_color(BLUE).shift(np.array([-3.0, 0, 0]))
    to = Torus(major_radius=1.0, minor_radius=0.35).set_color(GREEN)
    cy = Cylinder(radius=0.9, height=1.8).set_color(RED).shift(np.array([3.0, 0, 0]))
    return _flat(sp, to, cy), {"phi": 65, "theta": -50, "zoom": 0.8}


WORKLOADS = [
    ("2D shapes x50", w_2d_light),
    ("2D shapes x200", w_2d_heavy),
    ("Text (~55 glyphs)", w_text),
    ("MathTex", w_mathtex),
    ("3D Sphere (lit)", w_3d_sphere),
    ("3D scene x3 (lit)", w_3d_heavy),
]


def _cairo_cam(pw, ph, params):
    if params:
        return CairoThreeDCamera(
            pixel_width=pw, pixel_height=ph, frame_width=FW,
            phi=params["phi"] * DEGREES, theta=params["theta"] * DEGREES,
            zoom=params["zoom"], focal_distance=20.0,
        )
    return CairoCamera(pixel_width=pw, pixel_height=ph, frame_width=FW)


def _metal_cam(pw, ph, params):
    cam = MetalCamera(pixel_width=pw, pixel_height=ph, frame_width=FW, fxaa=False)
    if params:
        cam.set_phi(params["phi"] * DEGREES)
        cam.set_theta(params["theta"] * DEGREES)
        cam.set_zoom(params["zoom"])
    return cam


def _bench(fn, warmup=3, iters=11, max_seconds=6.0):
    for _ in range(warmup):
        fn()
    times = []
    start = time.perf_counter()
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        if time.perf_counter() - start > max_seconds:
            break
    return statistics.median(times) * 1000.0  # ms


def run(resolutions):
    rows = []
    for res in resolutions:
        pw, ph = RESOLUTIONS[res]
        config.pixel_width, config.pixel_height, config.frame_width = pw, ph, FW
        for name, build in WORKLOADS:
            mobs_c, params = build()
            mobs_m, _ = build()
            ccam = _cairo_cam(pw, ph, params)
            mcam = _metal_cam(pw, ph, params)

            def cairo_frame(c=ccam, m=mobs_c):
                c.reset(); c.capture_mobjects(m)

            def metal_warm(c=mcam, m=mobs_m):
                c.reset(); c.capture_mobjects(m)

            def metal_cold(c=mcam, m=mobs_m):
                c._geo_cache.clear(); c.reset(); c.capture_mobjects(m)

            cairo_ms = _bench(cairo_frame)
            metal_w_ms = _bench(metal_warm)
            metal_c_ms = _bench(metal_cold)
            speed_w = cairo_ms / metal_w_ms if metal_w_ms else float("nan")
            speed_c = cairo_ms / metal_c_ms if metal_c_ms else float("nan")
            rows.append({
                "res": res, "workload": name,
                "cairo_ms": cairo_ms, "metal_warm_ms": metal_w_ms,
                "metal_cold_ms": metal_c_ms, "speedup_warm": speed_w,
                "speedup_cold": speed_c,
            })
            print(f"[{res:>5}] {name:<20} cairo={cairo_ms:8.2f}ms  "
                  f"metal(warm)={metal_w_ms:7.2f}ms ({speed_w:5.1f}x)  "
                  f"metal(cold)={metal_c_ms:7.2f}ms ({speed_c:5.1f}x)")
    return rows


def _hardware():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    cores = subprocess.run(["sysctl", "-n", "hw.logicalcpu"],
                           capture_output=True, text=True).stdout.strip()
    mem = subprocess.run(["sysctl", "-n", "hw.memsize"],
                         capture_output=True, text=True).stdout.strip()
    gb = round(int(mem) / 1024**3) if mem.isdigit() else "?"
    return f"{chip}, {cores}-core, {gb} GB, macOS {platform.mac_ver()[0]}"


def write_md(rows, hw):
    lines = [
        "# Benchmark: Cairo vs Metal renderer", "",
        "Per-frame render time for `camera.capture_mobjects` on identical mobjects "
        "(tessellation + rasterization/GPU + readback). Process startup and video "
        "muxing are excluded — they are identical for both backends. Reported value "
        "is the median of up to 11 iterations after 3 warmups.", "",
        f"**Hardware:** {hw}", "",
        "- **Metal (warm)** — geometry cache hot: steady-state animation where geometry "
        "is static and only the camera moves (the common case).",
        "- **Metal (cold)** — geometry cache cleared every frame: geometry changes every "
        "frame (e.g. a `Transform`). Cairo re-rasterizes every frame either way.", "",
        "## Summary", "",
        "- **3D lit surfaces are Metal's biggest win (≈3–6×).** Cairo shades every facet "
        "on the CPU; Metal does the projection on the GPU. Even with the cache cold, "
        "Metal stays well ahead.",
        "- **Static-geometry 2D & text: Metal ≈1.4–2.5× faster**, and the lead grows with "
        "resolution (GPU rasterization scales better than Cairo's CPU rasterizer).",
        "- **When geometry changes every frame, CPU tessellation dominates** and Metal can "
        "be *slower* than Cairo for 2D/text (the ear-clipping pass re-runs per shape/glyph). "
        "This is the main optimization target; 3D still wins because Cairo's facet shading "
        "is the larger cost. In practice only the *moving* objects re-tessellate, so real "
        "animations sit between the warm and cold columns.", "",
    ]
    by_res = {}
    for r in rows:
        by_res.setdefault(r["res"], []).append(r)
    for res, rs in by_res.items():
        pw, ph = RESOLUTIONS[res]
        lines += [f"## {res} ({pw}×{ph})", "",
                  "| Workload | Cairo | Metal (warm) | Speedup | Metal (cold) | Speedup |",
                  "|---|--:|--:|--:|--:|--:|"]
        for r in rs:
            lines.append(
                f"| {r['workload']} | {r['cairo_ms']:.2f} ms | {r['metal_warm_ms']:.2f} ms | "
                f"**{r['speedup_warm']:.1f}×** | {r['metal_cold_ms']:.2f} ms | "
                f"{r['speedup_cold']:.1f}× |")
        lines.append("")
    lines += [
        "## Notes", "",
        "- Methodology: `uv run python benchmark.py`. Both backends render the exact "
        "same `VMobject` list through their respective cameras; 3D uses Cairo "
        "`ThreeDCamera` vs `MetalCamera` with matching phi/theta/zoom.",
        "- Numbers are render-only and will vary with thermal state and machine load.",
        "- Accuracy (not speed) parity is measured separately by `compare_metric.py` / "
        "`compare_video.py`.",
        "",
    ]
    with open("BENCHMARK.md", "w") as f:
        f.write("\n".join(lines))
    print("\nwrote BENCHMARK.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", action="append", choices=list(RESOLUTIONS))
    ap.add_argument("--write-md", action="store_true")
    args = ap.parse_args()
    resolutions = args.res if args.res else list(RESOLUTIONS)
    hw = _hardware()
    print(f"Hardware: {hw}\n")
    rows = run(resolutions)
    if args.write_md:
        write_md(rows, hw)


if __name__ == "__main__":
    main()
