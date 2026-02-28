"""3D hologram-inspired scenes for Metal renderer.

Inspired by 3Blue1Brown's "How are holograms possible?" video.
Adapted from ManimGL custom shader code to Manim CE VMobjects.

Usage:
    uv run manim render -qh --renderer metal demo_hologram.py WaveInterference
    uv run manim render -qh --renderer metal demo_hologram.py ZonePlate
    uv run manim render -qh --renderer metal demo_hologram.py DiffractionGrating
    uv run manim render -qh --renderer metal demo_hologram.py HologramFull
"""

from __future__ import annotations

import numpy as np
from manim import *

import manim_metal  # noqa: F401 — apply Metal renderer patches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wave_value(point, source, wave_number, time, decay=0.5):
    """Scalar wave cos(k*r - omega*t) / r^decay at a single point."""
    r = np.linalg.norm(point[:2] - source[:2])
    if r < 1e-6:
        return 1.0
    return np.cos(TAU * (wave_number * r - time)) * (r + 1) ** (-decay)


def make_wavefronts(source, n_rings=12, max_radius=6, color=BLUE, base_opacity=0.6):
    """Concentric circles representing wave-fronts from *source*."""
    rings = VGroup()
    for i in range(1, n_rings + 1):
        r = max_radius * i / n_rings
        ring = Circle(radius=r, color=color, stroke_width=2,
                      stroke_opacity=base_opacity * (1 - i / (n_rings + 1)))
        ring.move_to(source)
        rings.add(ring)
    return rings


# ---------------------------------------------------------------------------
# Scene 1 — Wave Interference (two point sources)
# ---------------------------------------------------------------------------

class WaveInterference(ThreeDScene):
    """Two point sources emitting spherical waves that interfere.

    Inspired by the hologram video's wave-interference visualisation.
    """

    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES)

        # --- Sources ---
        src_a = np.array([-2.0, 0.0, 0.0])
        src_b = np.array([2.0, 0.0, 0.0])

        dot_a = Dot3D(point=src_a, color=YELLOW, radius=0.08)
        dot_b = Dot3D(point=src_b, color=YELLOW, radius=0.08)
        label_a = Text("S₁", font_size=24, color=YELLOW).next_to(dot_a, UP, buff=0.15)
        label_b = Text("S₂", font_size=24, color=YELLOW).next_to(dot_b, UP, buff=0.15)

        # --- Wave-fronts (concentric rings) ---
        fronts_a = make_wavefronts(src_a, n_rings=14, color=BLUE_C)
        fronts_b = make_wavefronts(src_b, n_rings=14, color=RED_C)

        self.play(
            FadeIn(dot_a), FadeIn(dot_b),
            Write(label_a), Write(label_b),
            run_time=1,
        )

        # Expand wave-fronts outward
        for rings in (fronts_a, fronts_b):
            for r in rings:
                r.scale(0.01).set_stroke(opacity=0)

        self.add(fronts_a, fronts_b)

        anims = []
        for rings, opacity_base in ((fronts_a, 0.5), (fronts_b, 0.5)):
            for i, r in enumerate(rings):
                target_scale = 1 / 0.01
                anims.append(
                    r.animate.scale(target_scale).set_stroke(
                        opacity=opacity_base * (1 - i / (len(rings) + 1))
                    )
                )

        self.play(*anims, run_time=3, rate_func=linear)
        self.wait(0.5)

        # --- Interference pattern — grid of small dots coloured by amplitude ---
        n_grid = 30
        xs = np.linspace(-5, 5, n_grid)
        ys = np.linspace(-4, 4, n_grid)
        wave_k = 3.0
        time_val = 0.0

        dots = VGroup()
        for x in xs:
            for y in ys:
                pt = np.array([x, y, 0.0])
                v = (wave_value(pt, src_a, wave_k, time_val)
                     + wave_value(pt, src_b, wave_k, time_val))
                # Map amplitude [-2, 2] → colour
                t = np.clip((v + 2) / 4, 0, 1)
                col = interpolate_color(RED, BLUE, t)
                d = Dot(point=pt, color=col, radius=0.06, fill_opacity=0.8)
                dots.add(d)

        self.play(FadeIn(dots, lag_ratio=0.005), run_time=2)
        self.wait(0.5)

        # Camera sweep
        self.move_camera(phi=45 * DEGREES, theta=20 * DEGREES, run_time=3)
        self.wait(0.5)
        self.move_camera(phi=75 * DEGREES, theta=-120 * DEGREES, run_time=4)
        self.wait(1)


# ---------------------------------------------------------------------------
# Scene 2 — Zone Plate
# ---------------------------------------------------------------------------

class ZonePlate(ThreeDScene):
    """Fresnel zone plate — concentric rings of alternating transparency.

    A zone plate acts as a hologram of a single point source.
    """

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

        # --- Build the zone plate ---
        n_zones = 40
        max_r = 3.5
        zone_rings = VGroup()
        for n in range(1, n_zones + 1):
            r_inner = max_r * np.sqrt((n - 1) / n_zones)
            r_outer = max_r * np.sqrt(n / n_zones)
            ring = Annulus(
                inner_radius=r_inner,
                outer_radius=r_outer,
                color=WHITE if n % 2 == 0 else BLACK,
                fill_opacity=0.85 if n % 2 == 0 else 0.6,
                stroke_width=0,
            )
            zone_rings.add(ring)

        plate_label = Text("Zone Plate", font_size=28, color=WHITE)
        plate_label.next_to(zone_rings, UP, buff=0.3)

        # Animate zone plate appearing ring by ring
        self.play(
            LaggedStart(
                *(FadeIn(r, scale=0.8) for r in zone_rings),
                lag_ratio=0.03,
            ),
            run_time=3,
        )
        self.play(Write(plate_label), run_time=0.8)
        self.wait(0.5)

        # --- Show the focal point ---
        focal_z = 5.0
        focal_dot = Dot3D(point=[0, 0, focal_z], color=YELLOW, radius=0.1)
        focal_label = Text("Focus", font_size=24, color=YELLOW)
        focal_label.next_to(focal_dot, RIGHT, buff=0.15)

        # Converging rays from zone-plate edge to focus
        n_rays = 16
        rays = VGroup()
        for angle in np.linspace(0, TAU, n_rays, endpoint=False):
            edge = np.array([max_r * 0.7 * np.cos(angle),
                             max_r * 0.7 * np.sin(angle), 0.0])
            ray = Line(start=edge, end=[0, 0, focal_z],
                       color=YELLOW, stroke_width=1.5, stroke_opacity=0.6)
            rays.add(ray)

        self.play(
            FadeIn(focal_dot),
            Write(focal_label),
            LaggedStart(*(Create(r) for r in rays), lag_ratio=0.04),
            run_time=2,
        )
        self.wait(0.5)

        # Camera rotation to see 3D structure
        self.move_camera(phi=55 * DEGREES, theta=30 * DEGREES, run_time=3)
        self.wait(0.5)
        self.move_camera(phi=80 * DEGREES, theta=-90 * DEGREES, run_time=3)
        self.wait(1)


# ---------------------------------------------------------------------------
# Scene 3 — Diffraction Grating
# ---------------------------------------------------------------------------

class DiffractionGrating(ThreeDScene):
    """Plane wave hitting a diffraction grating → multiple beams.

    Inspired by the "FullDiffractionGrating" scene from 3b1b's video.
    """

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES)

        # --- Incoming plane-wave fronts ---
        n_fronts = 20
        fronts_in = VGroup()
        for i in range(n_fronts):
            x = -5.0 + i * 0.5
            line = Line(
                start=[x, -4, 0], end=[x, 4, 0],
                color=BLUE_B, stroke_width=1.5,
                stroke_opacity=0.6 * (1 - abs(i - n_fronts / 2) / (n_fronts / 2)),
            )
            fronts_in.add(line)

        incoming_label = Text("Plane wave", font_size=22, color=BLUE_B)
        incoming_label.move_to([-4, 3.5, 0])

        # --- Grating (vertical slits) ---
        n_slits = 8
        slit_spacing = 0.8
        grating_y = np.linspace(
            -slit_spacing * (n_slits - 1) / 2,
            slit_spacing * (n_slits - 1) / 2,
            n_slits,
        )

        grating_bars = VGroup()
        for i in range(n_slits + 1):
            y_lo = -slit_spacing * n_slits / 2 + i * slit_spacing - slit_spacing * 0.35
            y_hi = y_lo + slit_spacing * 0.7
            if i == 0:
                y_lo = -4
                y_hi = grating_y[0] - slit_spacing * 0.15
            elif i == n_slits:
                y_lo = grating_y[-1] + slit_spacing * 0.15
                y_hi = 4
            else:
                y_lo = grating_y[i - 1] + slit_spacing * 0.15
                y_hi = grating_y[i] - slit_spacing * 0.15
            bar = Rectangle(
                width=0.15, height=y_hi - y_lo,
                color=GREY_B, fill_opacity=0.9, stroke_width=0.5,
            )
            bar.move_to([0, (y_lo + y_hi) / 2, 0])
            grating_bars.add(bar)

        grating_label = Text("Grating", font_size=22, color=GREY_B)
        grating_label.move_to([0, -3.8, 0])

        # --- Outgoing spherical wave-fronts from each slit ---
        all_slit_waves = VGroup()
        for sy in grating_y:
            slit_pt = np.array([0.0, sy, 0.0])
            waves = make_wavefronts(slit_pt, n_rings=8, max_radius=5,
                                    color=RED_C, base_opacity=0.35)
            # Only show right half
            for ring in waves:
                ring.set_stroke(opacity=ring.get_stroke_opacity() * 0.5)
            all_slit_waves.add(waves)

        # --- Diffraction orders (bright beams) ---
        orders = VGroup()
        for m in [-2, -1, 0, 1, 2]:
            angle = np.arcsin(np.clip(m * 0.25, -0.9, 0.9))  # simplified
            end_pt = np.array([6 * np.cos(angle), 6 * np.sin(angle), 0.0])
            beam = Line(
                start=ORIGIN, end=end_pt,
                color=YELLOW, stroke_width=3 - abs(m) * 0.6,
                stroke_opacity=0.8,
            )
            order_label = Text(f"m={m}", font_size=18, color=YELLOW)
            order_label.next_to(beam.get_end(), RIGHT if m >= 0 else LEFT, buff=0.1)
            orders.add(VGroup(beam, order_label))

        # --- Animate ---
        self.play(
            LaggedStart(*(FadeIn(f) for f in fronts_in), lag_ratio=0.05),
            Write(incoming_label),
            run_time=2,
        )
        self.play(FadeIn(grating_bars), Write(grating_label), run_time=1)
        self.wait(0.5)

        self.play(
            LaggedStart(*(FadeIn(w) for w in all_slit_waves), lag_ratio=0.08),
            run_time=2,
        )
        self.wait(0.5)

        self.play(
            LaggedStart(*(GrowFromCenter(o) for o in orders), lag_ratio=0.1),
            run_time=2,
        )
        self.wait(0.5)

        # Camera sweep
        self.move_camera(phi=50 * DEGREES, theta=-20 * DEGREES, run_time=3)
        self.wait(0.5)
        self.move_camera(phi=70 * DEGREES, theta=-100 * DEGREES, run_time=4)
        self.wait(1)


# ---------------------------------------------------------------------------
# Scene 4 — Full Hologram Demo (combined)
# ---------------------------------------------------------------------------

class HologramFull(ThreeDScene):
    """Full hologram recording & playback demonstration.

    1. Object emits spherical waves.
    2. Reference beam (plane wave) arrives.
    3. Interference pattern recorded on film (zone plate-like).
    4. Playback: reference beam through film → reconstructed object wave.
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)

        # ---- Act 1: Object and reference waves ----
        obj_pos = np.array([-3.0, 0.0, 2.0])
        film_x = 0.0

        obj_dot = Dot3D(point=obj_pos, color=YELLOW, radius=0.1)
        obj_label = Text("Object", font_size=22, color=YELLOW)
        obj_label.next_to(obj_dot, UP + LEFT, buff=0.15)

        # Film (vertical rectangle)
        film = Rectangle(width=0.1, height=5, color=WHITE,
                         fill_opacity=0.15, stroke_width=1.5)
        film.move_to([film_x, 0, 0])
        film_label = Text("Film", font_size=22, color=WHITE)
        film_label.next_to(film, DOWN, buff=0.2)

        self.play(FadeIn(obj_dot), Write(obj_label), run_time=0.8)
        self.play(FadeIn(film), Write(film_label), run_time=0.8)

        # Object spherical wave-fronts
        obj_fronts = make_wavefronts(obj_pos[:2].tolist() + [obj_pos[2]],
                                     n_rings=10, max_radius=5, color=BLUE_C)
        # Move all rings to obj_pos z-level
        for r in obj_fronts:
            r.move_to([r.get_center()[0], r.get_center()[1], obj_pos[2]])

        obj_waves_label = Text("Object wave", font_size=20, color=BLUE_C)
        obj_waves_label.move_to([-3, 3.5, 0])

        self.play(FadeIn(obj_fronts, lag_ratio=0.05), Write(obj_waves_label), run_time=2)
        self.wait(0.5)

        # Reference plane wave
        ref_fronts = VGroup()
        for i in range(16):
            x = -6 + i * 0.4
            line = Line(start=[x, -3, -1], end=[x, 3, -1],
                        color=RED_B, stroke_width=1.5,
                        stroke_opacity=0.5)
            ref_fronts.add(line)

        ref_label = Text("Reference wave", font_size=20, color=RED_B)
        ref_label.move_to([-4, -3.5, 0])

        self.play(FadeIn(ref_fronts, lag_ratio=0.03), Write(ref_label), run_time=1.5)
        self.wait(0.5)

        # Camera move to see the setup
        self.move_camera(phi=55 * DEGREES, theta=-30 * DEGREES, run_time=2)
        self.wait(0.5)

        # ---- Act 2: Interference pattern on film ----
        title_interference = Text("Recording interference", font_size=26, color=GREEN)
        title_interference.to_edge(UP)

        # Zone-plate-like pattern on the film
        n_zones = 30
        zone_strips = VGroup()
        for n in range(n_zones):
            y_center = -2.5 + 5.0 * n / n_zones
            strip_h = 5.0 / n_zones
            # Interference → alternating bright/dark with frequency increasing from center
            brightness = 0.5 + 0.5 * np.cos(
                PI * 4.0 * ((y_center / 2.5) ** 2)
            )
            strip = Rectangle(
                width=0.12, height=strip_h,
                color=interpolate_color(BLACK, WHITE, brightness),
                fill_opacity=0.9,
                stroke_width=0,
            )
            strip.move_to([film_x, y_center, 0])
            zone_strips.add(strip)

        self.play(
            Write(title_interference),
            FadeOut(film),
            LaggedStart(*(FadeIn(s) for s in zone_strips), lag_ratio=0.02),
            run_time=2,
        )
        self.wait(1)

        # ---- Act 3: Playback — reference beam reconstructs object wave ----
        title_playback = Text("Playback: reconstructed wave", font_size=26, color=TEAL)
        title_playback.to_edge(UP)

        # Fade out object and its waves
        self.play(
            FadeOut(obj_fronts), FadeOut(obj_dot), FadeOut(obj_label),
            FadeOut(obj_waves_label),
            ReplacementTransform(title_interference, title_playback),
            run_time=1.5,
        )

        # Reconstructed spherical wave on the other side
        recon_fronts = make_wavefronts([film_x, 0, 0], n_rings=8,
                                       max_radius=4, color=TEAL_C)
        recon_label = Text("Reconstructed wave", font_size=20, color=TEAL_C)
        recon_label.move_to([3, 3.5, 0])

        # Virtual image dot
        virtual_dot = Dot3D(point=obj_pos, color=TEAL, radius=0.1)
        virtual_label = Text("Virtual image", font_size=20, color=TEAL)
        virtual_label.next_to(virtual_dot, UP + LEFT, buff=0.15)

        # Diverging rays from film suggesting virtual image behind
        div_rays = VGroup()
        for angle in np.linspace(-0.6, 0.6, 8):
            end = np.array([5 * np.cos(angle), 5 * np.sin(angle), 0.0])
            ray = DashedLine(
                start=[film_x, 0, 0], end=end,
                color=TEAL, stroke_width=1.5, stroke_opacity=0.5,
                dash_length=0.15,
            )
            div_rays.add(ray)

        self.play(
            FadeIn(recon_fronts, lag_ratio=0.05),
            Write(recon_label),
            run_time=2,
        )
        self.play(
            LaggedStart(*(Create(r) for r in div_rays), lag_ratio=0.05),
            FadeIn(virtual_dot), Write(virtual_label),
            run_time=2,
        )
        self.wait(0.5)

        # Grand camera sweep
        self.move_camera(phi=40 * DEGREES, theta=30 * DEGREES, run_time=3)
        self.wait(0.5)
        self.move_camera(phi=70 * DEGREES, theta=-120 * DEGREES, run_time=4)
        self.wait(0.5)

        # Final slow rotation
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(6)
        self.stop_ambient_camera_rotation()
        self.wait(1)
