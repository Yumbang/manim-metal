"""Benchmark: Metal vs Cairo renderer on a complex scene.

Usage:
    uv run python benchmark.py
"""

from __future__ import annotations

import math
import time

from manim import *

import manim_metal  # noqa: F401 — apply patches

# Shared scene config
PIXEL_W = 1920
PIXEL_H = 1080
FPS = 60


class StressTestScene(Scene):
    """A visually complex scene designed to stress the renderer.

    Contains:
    - 120 overlapping circles in a spiral with gradient fills + strokes
    - 80 rotating squares in a grid with varying opacity
    - A large polygon star burst (20-pointed star)
    - Sine-wave made of 200 line segments
    - Multiple concurrent Transform and FadeIn/FadeOut animations
    - A fractal-like recursive triangle pattern (Sierpinski-ish)
    """

    def construct(self):
        # ---- Layer 1: Spiral of gradient circles ----
        spiral_circles = VGroup()
        for i in range(120):
            angle = i * 0.3
            r = 0.05 * i
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            c = Circle(radius=0.15 + 0.005 * i)
            c.set_fill(
                color=ManimColor.from_hsv((i / 120, 0.8, 0.9)),
                opacity=0.4 + 0.3 * math.sin(i * 0.1),
            )
            c.set_stroke(
                color=ManimColor.from_hsv(((i / 120 + 0.5) % 1.0, 0.9, 1.0)),
                width=1.5 + math.sin(i * 0.2),
            )
            c.move_to([x, y, 0])
            spiral_circles.add(c)

        # ---- Layer 2: Grid of rotating squares ----
        grid_squares = VGroup()
        for row in range(8):
            for col in range(10):
                sq = Square(side_length=0.35)
                sq.set_fill(
                    color=ManimColor.from_hsv((row / 8, 0.6, 0.8)),
                    opacity=0.2 + 0.05 * col,
                )
                sq.set_stroke(WHITE, width=0.8)
                sq.move_to([
                    -6 + col * 1.35,
                    -3.5 + row * 1.0,
                    0,
                ])
                sq.rotate(col * 0.15 + row * 0.1)
                grid_squares.add(sq)

        # ---- Layer 3: Star burst polygon ----
        star_points = []
        for i in range(40):
            angle = i * TAU / 40
            r = 3.0 if i % 2 == 0 else 1.2
            star_points.append([r * math.cos(angle), r * math.sin(angle), 0])
        star = Polygon(*star_points)
        star.set_fill(YELLOW, opacity=0.15)
        star.set_stroke(GOLD, width=2)

        # ---- Layer 4: Sine wave from line segments ----
        sine_segments = VGroup()
        prev = None
        for i in range(200):
            x = -7 + i * 0.07
            y = 2.0 * math.sin(x * 1.5) + 0.5 * math.sin(x * 4.7)
            pt = [x, y, 0]
            if prev is not None:
                line = Line(prev, pt)
                line.set_stroke(
                    color=ManimColor.from_hsv((i / 200, 1.0, 1.0)),
                    width=2.5,
                )
                sine_segments.add(line)
            prev = pt

        # ---- Layer 5: Recursive triangles (depth 4 Sierpinski-ish) ----
        def sierpinski(center, size, depth):
            if depth == 0:
                tri = RegularPolygon(n=3, start_angle=PI / 2)
                tri.set_height(size)
                tri.move_to(center)
                tri.set_fill(
                    ManimColor.from_hsv((depth * 0.2, 0.7, 0.9)),
                    opacity=0.3,
                )
                tri.set_stroke(BLUE_C, width=0.8)
                return VGroup(tri)
            group = VGroup()
            half = size / 2
            offsets = [
                UP * half * 0.5,
                DOWN * half * 0.25 + LEFT * half * 0.43,
                DOWN * half * 0.25 + RIGHT * half * 0.43,
            ]
            for off in offsets:
                group.add(*sierpinski(center + off, half, depth - 1))
            return group

        sierpinski_group = sierpinski(ORIGIN, 5.0, 4)

        # ---- Animations ----
        # Phase 1: Build up all layers
        self.play(
            FadeIn(grid_squares, lag_ratio=0.02),
            Create(star, run_time=2),
            run_time=2,
        )

        self.play(
            *[Create(c, run_time=0.5) for c in spiral_circles[:30]],
            FadeIn(sine_segments, lag_ratio=0.01),
            run_time=2,
        )

        self.play(
            FadeIn(spiral_circles[30:], lag_ratio=0.02),
            FadeIn(sierpinski_group, lag_ratio=0.03),
            run_time=2,
        )

        # Phase 2: Transforms — rotate everything, color shifts
        self.play(
            Rotate(spiral_circles, angle=PI / 3, about_point=ORIGIN),
            Rotate(grid_squares, angle=-PI / 6, about_point=ORIGIN),
            star.animate.scale(1.3).set_fill(opacity=0.3),
            run_time=2,
        )

        # Phase 3: More chaos
        self.play(
            spiral_circles.animate.shift(RIGHT * 0.5),
            grid_squares.animate.shift(LEFT * 0.3),
            FadeOut(sine_segments, lag_ratio=0.01),
            sierpinski_group.animate.scale(0.6).shift(DOWN),
            run_time=2,
        )

        # Phase 4: Grand finale — everything fades
        self.play(
            FadeOut(spiral_circles, lag_ratio=0.01),
            FadeOut(grid_squares, lag_ratio=0.01),
            FadeOut(star),
            FadeOut(sierpinski_group, lag_ratio=0.02),
            run_time=2,
        )

        self.wait(0.5)


def run_benchmark(renderer_name: str) -> dict:
    """Run the stress test scene with a given renderer and return timing info."""
    config.pixel_width = PIXEL_W
    config.pixel_height = PIXEL_H
    config.frame_rate = FPS
    config.renderer = renderer_name
    config.disable_caching = True
    config.media_dir = f"./media_{renderer_name}"

    scene = StressTestScene()

    t0 = time.perf_counter()
    scene.render()
    elapsed = time.perf_counter() - t0

    num_plays = scene.renderer.num_plays
    output = scene.renderer.file_writer.movie_file_path

    return {
        "renderer": renderer_name,
        "elapsed_s": elapsed,
        "num_plays": num_plays,
        "output": str(output),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  manim-metal Benchmark: Metal vs Cairo")
    print(f"  Resolution: {PIXEL_W}x{PIXEL_H} @ {FPS}fps")
    print("=" * 70)
    print()

    results = []
    for renderer in ("cairo", "metal"):
        print(f"--- Running with {renderer.upper()} renderer ---")
        r = run_benchmark(renderer)
        results.append(r)
        print(f"  Time: {r['elapsed_s']:.2f}s | Animations: {r['num_plays']}")
        print(f"  Output: {r['output']}")
        print()

    # Summary
    cairo_t = results[0]["elapsed_s"]
    metal_t = results[1]["elapsed_s"]
    speedup = cairo_t / metal_t if metal_t > 0 else float("inf")

    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Cairo:  {cairo_t:.2f}s")
    print(f"  Metal:  {metal_t:.2f}s")
    print(f"  Speedup: {speedup:.2f}x {'(Metal faster)' if speedup > 1 else '(Cairo faster)'}")
    print("=" * 70)
