"""3Blue1Brown-style Gaussian integral scene — adapted for Manim CE + Metal.

Based on the BellCurveArea scene from 3b1b/videos (_2023/gauss_int/integral.py).

Usage:
    uv run manim render -ql demo_3b1b.py BellCurveArea           # Cairo
    uv run manim render -ql --renderer metal demo_3b1b.py BellCurveArea  # Metal
"""

from __future__ import annotations

import numpy as np
from manim import *

import manim_metal  # noqa: F401 — apply Metal renderer patches


class BellCurveArea(Scene):
    def construct(self):
        # --- Setup axes and bell curve ---
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 1.5, 0.5],
            x_length=14,
            y_length=5,
            axis_config={"include_numbers": True},
            tips=False,
        )
        axes.to_edge(DOWN)

        graph = axes.plot(lambda x: np.exp(-(x**2)), color=BLUE, stroke_width=3)

        graph_label = MathTex(r"e^{-x^2}", font_size=72)
        graph_label.set_color(BLUE)
        graph_label.next_to(graph.point_from_proportion(0.6), UR)

        self.add(axes)
        self.play(Create(graph))
        self.play(Write(graph_label))
        self.wait(0.5)

        # --- Show integral notation ---
        integral = MathTex(
            r"\int_{-\infty}^{\infty}",
            r"e^{-x^2}",
            r"\, dx",
            font_size=60,
        )
        integral.to_edge(UP)

        self.play(graph.animate.set_fill(BLUE, opacity=0.5))
        self.wait(0.5)
        self.play(
            Write(integral[0]),
            FadeTransform(graph_label.copy(), integral[1]),
        )
        self.play(TransformFromCopy(integral[1], integral[2]))
        self.wait(0.5)

        # --- Riemann rectangles ---
        colors = [BLUE_E, BLUE_D, TEAL_D, TEAL_E]
        rects = axes.get_riemann_rectangles(
            graph, x_range=[-4, 4], dx=0.2, color=colors
        )
        rects.set_stroke(WHITE, 1)
        rects.set_fill(opacity=0.75)

        # Highlight one rectangle
        rect = rects[len(rects) // 2 - 2].copy()
        rect.set_opacity(1)
        graph_label.set_stroke(BLACK, width=5, background=True)

        brace = Brace(rect, UP, buff=SMALL_BUFF)
        brace.set_stroke(BLACK, width=3, background=True)
        dx_label = brace.get_tex("dx", buff=SMALL_BUFF)
        dx_label.set_color(BLUE)

        self.play(
            FadeIn(rects, lag_ratio=0.1, run_time=3),
            graph.animate.set_fill(opacity=0),
            graph_label.animate.shift(SMALL_BUFF * UR),
        )
        self.wait(0.5)
        self.play(
            rects.animate.set_opacity(0.1),
            FadeIn(rect),
        )
        self.wait(0.5)
        self.play(
            graph_label.animate.scale(0.6).next_to(rect, LEFT, SMALL_BUFF)
        )
        self.play(Circumscribe(integral[1], time_width=1, run_time=1.5))
        self.wait(0.25)
        self.play(
            GrowFromCenter(brace),
            FadeIn(dx_label, shift=0.5 * UP),
        )
        self.play(Circumscribe(integral[2], time_width=1, run_time=1.5))
        self.wait(0.5)

        # --- Show summation (highlight each rectangle) ---
        rects.set_fill(opacity=0.8)
        rects.set_stroke(WHITE, 1)
        self.play(
            graph_label.animate.scale(1.4).next_to(
                graph.point_from_proportion(0.4), UL
            ),
            rects.animate.set_opacity(0.75),
            FadeOut(rect),
        )
        self.wait(0.25)
        self.play(
            LaggedStart(
                *(
                    r.animate.shift(0.25 * UP)
                    .set_color(YELLOW)
                    .set_rate_func(there_and_back)
                    for r in rects
                ),
                run_time=5,
                lag_ratio=0.1,
            ),
        )
        self.wait(0.5)

        # --- Shrinking dx — the iconic 3b1b moment ---
        for dx in [0.1, 0.075, 0.05, 0.03, 0.02, 0.01, 0.005]:
            new_rects = axes.get_riemann_rectangles(
                graph, x_range=[-4, 4], dx=dx, color=colors
            )
            new_rects.set_stroke(WHITE, 1)
            new_rects.set_fill(opacity=0.7)
            # Build a fresh brace for the new dx width
            new_rect_sample = new_rects[len(new_rects) // 2]
            new_brace = Brace(new_rect_sample, UP, buff=SMALL_BUFF)
            new_brace.set_stroke(BLACK, width=3, background=True)
            new_dx_label = new_brace.get_tex("dx", buff=SMALL_BUFF)
            new_dx_label.set_color(BLUE)
            self.play(
                Transform(rects, new_rects),
                Transform(brace, new_brace),
                Transform(dx_label, new_dx_label),
                run_time=0.8,
            )
        self.add(graph)
        self.play(
            FadeOut(brace),
            FadeOut(dx_label),
            Create(graph),
        )
        self.wait(0.5)

        # --- The big reveal: sqrt(pi) ---
        equals = MathTex("=", font_size=60)
        equals.next_to(integral, RIGHT)
        answer = MathTex(r"\sqrt{\pi}", font_size=72, color=YELLOW)
        answer.next_to(equals, RIGHT)

        answer_box = SurroundingRectangle(answer, buff=MED_SMALL_BUFF)
        answer_box.set_stroke(TEAL, 2)

        self.play(
            FadeIn(equals, shift=RIGHT * 0.3),
        )
        self.play(
            Write(answer),
            Create(answer_box),
            run_time=1.5,
        )
        self.wait(0.5)

        # --- Final dramatic flash ---
        self.play(
            Indicate(answer, scale_factor=1.3, color=YELLOW),
            run_time=1.5,
        )
        self.wait(1)

        # --- "Impossible" antiderivative note ---
        impossible = Text("No closed-form antiderivative!", font_size=36, color=RED)
        impossible.next_to(answer_box, DOWN, buff=0.8)

        arrow_down = Arrow(
            answer_box.get_bottom(),
            impossible.get_top(),
            color=RED,
        )
        self.play(
            GrowFromCenter(arrow_down),
            FadeIn(impossible, shift=DOWN * 0.3, scale=0.7),
        )
        self.wait(2)
