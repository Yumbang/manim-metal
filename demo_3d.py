"""3D stress test scenes for Metal renderer.

Usage:
    uv run manim render -ql --renderer metal demo_3d.py Axes3DStress
    uv run manim render -ql --renderer metal demo_3d.py DepthOrderTest
    uv run manim render -ql --renderer metal demo_3d.py CameraRotation
"""

from __future__ import annotations

import numpy as np
from manim import *

import manim_metal  # noqa: F401 — apply Metal renderer patches


class Axes3DStress(ThreeDScene):
    """Render 3D axes with multiple curves and shapes at various orientations."""

    def construct(self):
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-3, 3, 1],
            x_length=8,
            y_length=8,
            z_length=6,
        )

        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        # Parametric curve — helix
        helix = ParametricFunction(
            lambda t: np.array([np.cos(t), np.sin(t), 0.15 * t]),
            t_range=[-4 * PI, 4 * PI, 0.05],
            color=YELLOW,
            stroke_width=3,
        )

        # Another curve — spiral on XZ plane
        spiral = ParametricFunction(
            lambda t: np.array([0.3 * t * np.cos(t), 0, 0.3 * t * np.sin(t)]),
            t_range=[0, 4 * PI, 0.05],
            color=RED,
            stroke_width=2,
        )

        # Many circles at different z levels
        circles = VGroup()
        for z in np.linspace(-2, 2, 9):
            c = Circle(radius=0.5, color=BLUE, stroke_width=2, fill_opacity=0.3)
            c.move_to([0, 0, z])
            circles.add(c)

        # Grid of small squares
        squares = VGroup()
        for x in np.linspace(-3, 3, 7):
            for y in np.linspace(-3, 3, 7):
                sq = Square(side_length=0.3, color=GREEN, fill_opacity=0.4, stroke_width=1)
                sq.move_to([x, y, 0])
                squares.add(sq)

        self.add(axes)
        self.play(Create(helix), run_time=2)
        self.play(Create(spiral), run_time=1.5)
        self.play(FadeIn(circles, lag_ratio=0.1), run_time=1.5)
        self.play(FadeIn(squares, lag_ratio=0.02), run_time=2)
        self.wait(0.5)

        # Rotate camera
        self.move_camera(phi=60 * DEGREES, theta=30 * DEGREES, run_time=3)
        self.wait(0.5)
        self.move_camera(phi=30 * DEGREES, theta=-120 * DEGREES, run_time=3)
        self.wait(1)


class DepthOrderTest(ThreeDScene):
    """Test depth ordering — overlapping planes at different z values."""

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES)

        # Three overlapping rectangles at different depths
        r1 = Rectangle(width=3, height=2, color=RED, fill_opacity=0.7, stroke_width=2)
        r1.move_to([0, 0, -1])

        r2 = Rectangle(width=3, height=2, color=GREEN, fill_opacity=0.7, stroke_width=2)
        r2.move_to([0.5, 0.5, 0])

        r3 = Rectangle(width=3, height=2, color=BLUE, fill_opacity=0.7, stroke_width=2)
        r3.move_to([1, 1, 1])

        # Labels
        l1 = Text("z=-1", font_size=24, color=RED)
        l1.move_to(r1.get_center())
        l2 = Text("z=0", font_size=24, color=GREEN)
        l2.move_to(r2.get_center())
        l3 = Text("z=1", font_size=24, color=BLUE)
        l3.move_to(r3.get_center())

        self.add(r1, r2, r3, l1, l2, l3)
        self.wait(1)

        # Rotate to see depth ordering
        self.move_camera(phi=45 * DEGREES, theta=45 * DEGREES, run_time=2)
        self.wait(0.5)
        self.move_camera(phi=85 * DEGREES, theta=-30 * DEGREES, run_time=2)
        self.wait(1)


class CameraRotation(ThreeDScene):
    """Continuous ambient camera rotation around a 3D scene."""

    def construct(self):
        axes = ThreeDAxes(x_length=6, y_length=6, z_length=4)
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        # Star pattern — many lines through origin
        lines = VGroup()
        for angle in np.linspace(0, 2 * PI, 24, endpoint=False):
            line = Line(
                start=2 * np.array([np.cos(angle), np.sin(angle), 0]),
                end=2 * np.array([-np.cos(angle), -np.sin(angle), 0]),
                color=interpolate_color(BLUE, RED, angle / (2 * PI)),
                stroke_width=2,
            )
            lines.add(line)

        # Vertical lines
        for x in np.linspace(-2, 2, 5):
            for y in np.linspace(-2, 2, 5):
                vline = Line(
                    start=[x, y, -1.5],
                    end=[x, y, 1.5],
                    color=YELLOW,
                    stroke_width=1,
                )
                lines.add(vline)

        self.add(axes, lines)
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(8)
        self.stop_ambient_camera_rotation()
        self.wait(0.5)
