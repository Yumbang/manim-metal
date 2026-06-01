# ruff: noqa
"""Animated scenes for Cairo-vs-Metal VIDEO parity testing.

Each scene exercises a different aspect of animation that a single static
frame cannot: interpolated geometry tessellation, text animation, camera
motion, object rotation, opacity-over-time, depth ordering changes.

Rendered by compare_video.py with both --renderer cairo and --renderer metal.
"""

import manim_metal  # noqa: F401  (harmless under cairo; activates metal patches)
from manim import *


class VTransform(Scene):
    """Shape morph — per-frame tessellation of interpolated geometry."""

    def construct(self):
        c = Circle(radius=1.5).set_fill(BLUE, 0.7).set_stroke(WHITE, 4)
        s = Square(side_length=2.6).set_fill(RED, 0.7).set_stroke(YELLOW, 4)
        self.add(c)
        self.play(Transform(c, s), run_time=1.0)
        self.play(Transform(c, Triangle().scale(2).set_fill(GREEN, 0.7).set_stroke(WHITE, 4)),
                  run_time=1.0)


class VText(Scene):
    """Text + MathTex animation — earcut fills under Write/shift."""

    def construct(self):
        t = Text("manim-metal", color=YELLOW).scale(1.1).to_edge(UP)
        eq = MathTex(r"e^{i\pi} + 1 = 0", color=WHITE).scale(1.6)
        self.play(Write(t), run_time=1.0)
        self.play(FadeIn(eq, shift=UP), run_time=1.0)
        self.play(eq.animate.shift(2 * LEFT).scale(0.7), run_time=1.0)


class VShapes2D(Scene):
    """Multiple 2D shapes: create, move, rotate, fade — depth + opacity over time."""

    def construct(self):
        c = Circle(radius=1.0).set_fill(BLUE, 0.6).set_stroke(WHITE, 3).shift(3 * LEFT)
        s = Square(side_length=1.6).set_fill(RED, 0.5).set_stroke(WHITE, 3)
        p = RegularPolygon(5).scale(1.2).set_fill(GREEN, 0.7).set_stroke(ORANGE, 3).shift(3 * RIGHT)
        self.play(Create(c), Create(s), Create(p), run_time=1.0)
        self.play(Rotate(s, PI / 2), c.animate.shift(2 * RIGHT), p.animate.set_opacity(0.3),
                  run_time=1.0)


class VThreeDRot(ThreeDScene):
    """Lit sphere + torus with camera motion — shading consistency under camera animation."""

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        sphere = Sphere(radius=1.3).set_color(BLUE).shift(2 * LEFT)
        torus = Torus(major_radius=1.0, minor_radius=0.35).set_color(GREEN).shift(2 * RIGHT)
        self.add(sphere, torus)
        self.move_camera(theta=15 * DEGREES, run_time=2.0)


class VObjRotate(ThreeDScene):
    """Rotating 3D surface — facet normals change every frame (object rotation)."""

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-60 * DEGREES)
        cyl = Cylinder(radius=1.0, height=2.2).set_color(RED)
        self.add(cyl)
        self.play(Rotate(cyl, angle=PI, axis=RIGHT, run_time=2.0))
