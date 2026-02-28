"""Manim CE 3D compatibility test — all official 3D doc examples.

Renders every ThreeDScene example from Manim CE's documentation through
the Metal renderer to verify full compatibility.

Usage:
    uv run manim render -ql --renderer metal test_manim_ce_3d_compat.py <SceneName>
    uv run manim render -ql --renderer metal -a test_manim_ce_3d_compat.py  # all scenes
"""

from __future__ import annotations

import numpy as np
from manim import *

import manim_metal  # noqa: F401 — apply Metal renderer patches


# =============================================================================
# three_dimensions.py examples
# =============================================================================


class ParaSurface(ThreeDScene):
    """Parametric surface example from Surface docstring."""

    def func(self, u, v):
        return np.array([np.cos(u) * np.cos(v), np.cos(u) * np.sin(v), u])

    def construct(self):
        axes = ThreeDAxes(x_range=[-4, 4], x_length=8)
        surface = Surface(
            lambda u, v: axes.c2p(*self.func(u, v)),
            u_range=[-PI, PI],
            v_range=[0, TAU],
            resolution=8,
        )
        self.set_camera_orientation(theta=70 * DEGREES, phi=75 * DEGREES)
        self.add(axes, surface)


class FillByValueExample(ThreeDScene):
    """Surface with value-based coloring from Surface.set_fill_by_value docstring."""

    def construct(self):
        resolution_fa = 8
        self.set_camera_orientation(phi=75 * DEGREES, theta=-160 * DEGREES)
        axes = ThreeDAxes(
            x_range=(0, 5, 1), y_range=(0, 5, 1), z_range=(-1, 1, 0.5)
        )

        def param_surface(u, v):
            x = u
            y = v
            z = np.sin(x) * np.cos(y)
            return z

        surface_plane = Surface(
            lambda u, v: axes.c2p(u, v, param_surface(u, v)),
            resolution=(resolution_fa, resolution_fa),
            v_range=[0, 5],
            u_range=[0, 5],
        )
        surface_plane.set_style(fill_opacity=1)
        surface_plane.set_fill_by_value(
            axes=axes,
            colorscale=[(RED, -0.5), (YELLOW, 0), (GREEN, 0.5)],
            axis=2,
        )
        self.add(axes, surface_plane)


class ExampleSphere(ThreeDScene):
    """Multiple colored spheres from Sphere docstring."""

    def construct(self):
        self.set_camera_orientation(phi=PI / 6, theta=PI / 6)
        sphere1 = Sphere(
            center=(3, 0, 0),
            radius=1,
            resolution=(20, 20),
            u_range=[0.001, PI - 0.001],
            v_range=[0, TAU],
        )
        sphere1.set_color(RED)
        self.add(sphere1)
        sphere2 = Sphere(center=(-1, -3, 0), radius=2, resolution=(18, 18))
        sphere2.set_color(GREEN)
        self.add(sphere2)
        sphere3 = Sphere(center=(-1, 2, 0), radius=2, resolution=(16, 16))
        sphere3.set_color(BLUE)
        self.add(sphere3)


class Dot3DExample(ThreeDScene):
    """3D dots from Dot3D docstring."""

    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        axes = ThreeDAxes()
        dot_1 = Dot3D(point=axes.coords_to_point(0, 0, 1), color=RED)
        dot_2 = Dot3D(
            point=axes.coords_to_point(2, 0, 0), radius=0.1, color=BLUE
        )
        dot_3 = Dot3D(point=[0, 0, 0], radius=0.1, color=ORANGE)
        self.add(axes, dot_1, dot_2, dot_3)


class CubeExample(ThreeDScene):
    """Cube from Cube docstring."""

    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        axes = ThreeDAxes()
        cube = Cube(side_length=3, fill_opacity=0.7, fill_color=BLUE)
        self.add(cube)


class ExamplePrism(ThreeDScene):
    """Prism from Prism docstring."""

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=150 * DEGREES)
        prismSmall = Prism(dimensions=[1, 2, 3]).rotate(PI / 2)
        prismLarge = Prism(dimensions=[1.5, 3, 4.5]).move_to([2, 0, 0])
        self.add(prismSmall, prismLarge)


class ExampleCone(ThreeDScene):
    """Cone from Cone docstring."""

    def construct(self):
        axes = ThreeDAxes()
        cone = Cone(direction=X_AXIS + Y_AXIS + 2 * Z_AXIS, resolution=8)
        self.set_camera_orientation(phi=5 * PI / 11, theta=PI / 9)
        self.add(axes, cone)


class ExampleCylinder(ThreeDScene):
    """Cylinder from Cylinder docstring."""

    def construct(self):
        axes = ThreeDAxes()
        cylinder = Cylinder(radius=2, height=3)
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes, cylinder)


class ExampleLine3D(ThreeDScene):
    """Line3D from Line3D docstring."""

    def construct(self):
        axes = ThreeDAxes()
        line = Line3D(start=np.array([0, 0, 0]), end=np.array([2, 2, 2]))
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes, line)


class ParallelLineExample(ThreeDScene):
    """Parallel line from Line3D.parallel_to docstring."""

    def construct(self):
        self.set_camera_orientation(PI / 3, -PI / 4)
        ax = ThreeDAxes((-5, 5), (-5, 5), (-5, 5), 10, 10, 10)
        line1 = Line3D(RIGHT * 2, UP + OUT, color=RED)
        line2 = Line3D.parallel_to(line1, color=YELLOW)
        self.add(ax, line1, line2)


class PerpLineExample(ThreeDScene):
    """Perpendicular line from Line3D.perpendicular_to docstring."""

    def construct(self):
        self.set_camera_orientation(PI / 3, -PI / 4)
        ax = ThreeDAxes((-5, 5), (-5, 5), (-5, 5), 10, 10, 10)
        line1 = Line3D(RIGHT * 2, UP + OUT, color=RED)
        line2 = Line3D.perpendicular_to(line1, color=BLUE)
        self.add(ax, line1, line2)


class ExampleArrow3D(ThreeDScene):
    """Arrow3D from Arrow3D docstring."""

    def construct(self):
        axes = ThreeDAxes()
        arrow = Arrow3D(
            start=np.array([0, 0, 0]), end=np.array([2, 2, 2]), resolution=8
        )
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes, arrow)


class ExampleTorus(ThreeDScene):
    """Torus from Torus docstring."""

    def construct(self):
        axes = ThreeDAxes()
        torus = Torus()
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes, torus)


# =============================================================================
# polyhedra.py examples
# =============================================================================


class SquarePyramidScene(ThreeDScene):
    """Custom polyhedron from Polyhedron docstring."""

    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        vertex_coords = [
            [1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
            [-1, 1, 0],
            [0, 0, 2],
        ]
        faces_list = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 1, 2, 3]]
        pyramid = Polyhedron(vertex_coords, faces_list)
        self.add(pyramid)


class PolyhedronSubMobjects(ThreeDScene):
    """Polyhedron submobject access from docstring."""

    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        octahedron = Octahedron(edge_length=3)
        octahedron.graph[0].set_color(RED)
        octahedron.faces[2].set_color(YELLOW)
        self.add(octahedron)


class TetrahedronScene(ThreeDScene):
    """Tetrahedron from Tetrahedron docstring."""

    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        obj = Tetrahedron()
        self.add(obj)


class OctahedronScene(ThreeDScene):
    """Octahedron from Octahedron docstring."""

    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        obj = Octahedron()
        self.add(obj)


class IcosahedronScene(ThreeDScene):
    """Icosahedron from Icosahedron docstring."""

    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        obj = Icosahedron()
        self.add(obj)


class DodecahedronScene(ThreeDScene):
    """Dodecahedron from Dodecahedron docstring."""

    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        obj = Dodecahedron()
        self.add(obj)


class ConvexHull3DExample(ThreeDScene):
    """Convex hull from ConvexHull3D docstring."""

    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        points = [
            [1.93192757, 0.44134585, -1.52407061],
            [-0.93302521, 1.23206983, 0.64117067],
            [-0.44350918, -0.61043677, 0.21723705],
            [-0.42640268, -1.05260843, 1.61266094],
            [-1.84449637, 0.91238739, -1.85172623],
            [1.72068132, -0.11880457, 0.51881751],
            [0.41904805, 0.44938012, -1.86440686],
            [0.83864666, 1.66653337, 1.88960123],
            [0.22240514, -0.80986286, 1.34249326],
            [-1.29585759, 1.01516189, 0.46187522],
            [1.7776499, -1.59550796, -1.70240747],
            [0.80065226, -0.12530398, 1.70063977],
            [1.28960948, -1.44158255, 1.39938582],
            [-0.93538943, 1.33617705, -0.24852643],
            [-1.54868271, 1.7444399, -0.46170734],
        ]
        hull = ConvexHull3D(
            *points,
            faces_config={"stroke_opacity": 0},
            graph_config={
                "vertex_type": Dot3D,
                "edge_config": {
                    "stroke_color": BLUE,
                    "stroke_width": 2,
                    "stroke_opacity": 0.05,
                },
            },
        )
        dots = VGroup(*[Dot3D(point) for point in points])
        self.add(hull, dots)


# =============================================================================
# coordinate_systems.py examples
# =============================================================================


class PlotSurfaceExample(ThreeDScene):
    """Surface plot from ThreeDAxes.plot_surface docstring."""

    def construct(self):
        resolution_fa = 16
        self.set_camera_orientation(phi=75 * DEGREES, theta=-60 * DEGREES)
        axes = ThreeDAxes(
            x_range=(-3, 3, 1), y_range=(-3, 3, 1), z_range=(-5, 5, 1)
        )

        def param_trig(u, v):
            x = u
            y = v
            z = 2 * np.sin(x) + 2 * np.cos(y)
            return z

        trig_plane = axes.plot_surface(
            param_trig,
            resolution=(resolution_fa, resolution_fa),
            u_range=(-3, 3),
            v_range=(-3, 3),
            colorscale=[BLUE, GREEN, YELLOW, ORANGE, RED],
        )
        self.add(axes, trig_plane)


class GetYAxisLabelExample(ThreeDScene):
    """Y-axis label from ThreeDAxes.get_y_axis_label docstring."""

    def construct(self):
        ax = ThreeDAxes()
        lab = ax.get_y_axis_label(Tex("$y$-label"))
        self.set_camera_orientation(phi=2 * PI / 5, theta=PI / 5)
        self.add(ax, lab)


class GetZAxisLabelExample(ThreeDScene):
    """Z-axis label from ThreeDAxes.get_z_axis_label docstring."""

    def construct(self):
        ax = ThreeDAxes()
        lab = ax.get_z_axis_label(Tex("$z$-label"))
        self.set_camera_orientation(phi=2 * PI / 5, theta=PI / 5)
        self.add(ax, lab)


class GetAxisLabelsExample(ThreeDScene):
    """All axis labels from ThreeDAxes.get_axis_labels docstring."""

    def construct(self):
        self.set_camera_orientation(phi=2 * PI / 5, theta=PI / 5)
        axes = ThreeDAxes()
        labels = axes.get_axis_labels(
            Text("x-axis").scale(0.7),
            Text("y-axis").scale(0.45),
            Text("z-axis").scale(0.45),
        )
        self.add(axes, labels)


# =============================================================================
# Camera animation examples
# =============================================================================


class CameraMoveExample(ThreeDScene):
    """Camera movement with move_camera and ambient rotation."""

    def construct(self):
        axes = ThreeDAxes()
        cube = Cube(side_length=2, fill_opacity=0.5, fill_color=BLUE)
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.add(axes, cube)
        self.move_camera(phi=45 * DEGREES, theta=45 * DEGREES, run_time=2)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        self.wait(0.5)
