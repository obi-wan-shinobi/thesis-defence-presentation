"""Lead-in animation: an error "bowl" whose principal axes are TILTED relative to
the x / y coordinates you were handed. Gradient descent in x / y therefore takes
a curved route (a step in x bleeds into y: the coordinates are coupled). The view
then tilts overhead and the bowl flattens onto its own x / y axes, so the tilted
contour rings and the curved path stay visible on the contour map.

This sets up the keystone slide, which rotates to the bowl's own (eigen) axes and
shows the dynamics decoupling. So this slide stays pre-lambda: no eigenvalue /
rate vocabulary, no lambda symbol. The steep / shallow arrows are drawn faintly
along the diagonal principal directions only as a teaser that the "good" axes are
not x and y.

Render:
    manim -qh manim/src/bowl_to_contours.py BowlToContours
    manim -qh -s manim/src/bowl_to_contours.py BowlToContours

LIGHT-BACKGROUND TRAP (see theme.py): Manim assumes a black canvas, so every
surface, dot, line, arrow, axis, and label colour below is set explicitly from
the deck palette; nothing is left to default or it washes out on #F8FAFC.

CAMERA NOTE: phi is the polar angle measured FROM the vertical (z) axis, so
phi = 0 is straight top-down and phi = 90 deg is a pure side view. The intro
uses a moderate phi (more from above than side), and the final view keeps a
gentle tilt (NOT flat top-down) so the contours read as the bowl flattened onto
its original axes. Tune PHI_INTRO / PHI_FINAL / THETA_* below to re-aim.
"""

import numpy as np
from theme import AXIS_GRAY, BG, BLUE, CYAN, FG, ORANGE

from manim import (
    DEGREES,
    Arrow,
    Dot3D,
    Ellipse,
    FadeIn,
    FadeOut,
    Line,
    ReplacementTransform,
    Surface,
    Text,
    ThreeDScene,
    VGroup,
)

# Darker cyan for the thin wireframe so the bowl's shape reads against the
# semi-transparent cyan fill on the light background.
DARKCYAN = "#0077A8"

# ----------------------------------------------------------------------
# CAMERA ANGLES (edit these to re-aim the shot).
#   phi   = tilt from straight-down (0 = top-down, 90 = side-on)
#   theta = in-plane azimuth (-90 puts x to the right, y up/away)
# ----------------------------------------------------------------------
# Intro starts a touch more side-on, then orbits in the x, y plane and lifts to
# PHI_DESCENT so the concave inside of the bowl is revealed before the descent.
PHI_INTRO, THETA_INTRO = 45 * DEGREES, -135 * DEGREES
PHI_DESCENT, THETA_DESCENT = 30 * DEGREES, -110 * DEGREES
PHI_FINAL, THETA_FINAL = 15 * DEGREES, -90 * DEGREES  # gentle tilt, not flat

# ----------------------------------------------------------------------
# THE TILTED ERROR BOWL.
# In its OWN (principal) coordinates u, v the bowl is a clean paraboloid
# z = MU_SHALLOW u^2 + MU_STEEP v^2, with u the shallow/long axis (small
# curvature) and v the steep/short axis (large curvature). Those principal
# axes are rotated by THETA_TILT relative to the x, y axes, so in x, y the
# quadratic has a cross term: the coordinates are COUPLED, and the contour
# ellipses come out tilted. A moderate MU ratio keeps the descent a visible
# 2-D mix (the steep part does not die so fast that the path snaps onto one
# eigendirection).
# ----------------------------------------------------------------------
THETA_TILT = 30 * DEGREES
MU_SHALLOW = 0.30  # long principal axis (u): drops slowly
MU_STEEP = 0.62  # short principal axis (v): drops fast

X_HALF, Y_HALF = 2.8, 2.4

FRAME_CENTER = np.array([0.0, 0.0, 1.2])

_C, _S = float(np.cos(THETA_TILT)), float(np.sin(THETA_TILT))


def to_uv(x, y):
    """x, y (handed coordinates) -> u, v (the bowl's own principal coordinates)."""
    return x * _C + y * _S, -x * _S + y * _C


def to_xy(u, v):
    """u, v (principal) -> x, y (handed coordinates)."""
    return u * _C - v * _S, u * _S + v * _C


def bowl_z(x, y):
    u, v = to_uv(x, y)
    return MU_SHALLOW * u * u + MU_STEEP * v * v


class BowlToContours(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG

        # ------------------------------------------------------------------
        # The handed axes the bowl lives in: faint slate lines through the
        # origin (x, y in the base plane; z = height). These stay x and y; the
        # bowl is tilted RELATIVE to them. Drawn in raw world coordinates so
        # they line up exactly with the surface and contours.
        # ------------------------------------------------------------------
        x_axis = Line([-3.5, 0, 0], [3.5, 0, 0], color=AXIS_GRAY, stroke_width=1.8)
        y_axis = Line([0, -3.0, 0], [0, 3.0, 0], color=AXIS_GRAY, stroke_width=1.8)
        z_axis = Line([0, 0, 0], [0, 0, 4.0], color=AXIS_GRAY, stroke_width=1.8)
        z_axis.set_opacity(0.55)
        axes = VGroup(x_axis, y_axis, z_axis)

        # Axis labels stay upright (face the camera) so they read at any tilt.
        # The vertical z line is left unlabelled to keep the top-centre clear.
        x_label = Text("x", color=AXIS_GRAY, font_size=24).move_to([3.7, 0, 0])
        y_label = Text("y", color=AXIS_GRAY, font_size=24).move_to([0, 3.2, 0])

        # ------------------------------------------------------------------
        # The tilted bowl surface: semi-transparent cyan fill + thin dark-cyan
        # grid. Parametrised over the x, y rectangle, with z from the tilted
        # quadratic, so the surface itself is visibly skewed.
        # ------------------------------------------------------------------
        surface = Surface(
            lambda x, y: np.array([x, y, bowl_z(x, y)]),
            u_range=[-X_HALF, X_HALF],
            v_range=[-Y_HALF, Y_HALF],
            resolution=(28, 22),
            fill_opacity=0.7,
            checkerboard_colors=[CYAN, CYAN],
        )
        surface.set_stroke(color=DARKCYAN, width=0.45, opacity=0.45)

        # ------------------------------------------------------------------
        # Nested contour ellipses on the base plane (z = 0): TILTED by
        # THETA_TILT. Each level c gives semi-axes a = sqrt(c/MU_SHALLOW)
        # along u (long) and b = sqrt(c/MU_STEEP) along v (short), drawn
        # axis-aligned then rotated into the principal frame.
        # ------------------------------------------------------------------
        levels = [0.35, 0.8, 1.35, 2.0, 2.7]
        contours = VGroup()
        for c in levels:
            a = float(np.sqrt(c / MU_SHALLOW))
            b = float(np.sqrt(c / MU_STEEP))
            ring = Ellipse(width=2 * a, height=2 * b, color=CYAN, stroke_width=2.4)
            ring.set_fill(opacity=0.0)
            ring.rotate(THETA_TILT)
            contours.add(ring)
        contours.set_z(0.0)

        # ------------------------------------------------------------------
        # Stage 1: camera slightly from above; fade the axes + bowl in.
        # ------------------------------------------------------------------
        self.set_camera_orientation(
            phi=PHI_INTRO, theta=THETA_INTRO, zoom=0.78, frame_center=FRAME_CENTER
        )
        self.add_fixed_orientation_mobjects(x_label, y_label)
        self.play(
            FadeIn(surface),
            FadeIn(axes),
            FadeIn(x_label),
            FadeIn(y_label),
            run_time=1.3,
        )
        # Orbit in the x, y plane (and lift a little) so the concave inside of
        # the bowl is revealed before the descent starts.
        self.move_camera(
            phi=PHI_DESCENT,
            theta=THETA_DESCENT,
            run_time=2.4,
            zoom=0.78,
            frame_center=FRAME_CENTER,
        )
        self.wait(0.4)

        # ------------------------------------------------------------------
        # Stage 2: TRUE gradient descent. Each step is along -gradient, which
        # is perpendicular to the (tilted) contour and does NOT point at the
        # minimum, so the path CURVES: the steep (v) part dies first and the
        # route bends toward the shallow (u) direction. The start is generic
        # (off both principal axes) so the route is a real 2-D mix. The step
        # is small enough that each component decays monotonically (no zigzag,
        # no overshoot).
        # ------------------------------------------------------------------
        eta = 0.5
        fac_u = 1.0 - eta * 2.0 * MU_SHALLOW  # per-step shrink along u (slow)
        fac_v = 1.0 - eta * 2.0 * MU_STEEP  # per-step shrink along v (fast)
        n_steps = 6
        # Generic start: high up and off the x axis, with large components along
        # BOTH principal axes, so the route is a clear 2-D mix and ends along
        # the (diagonal) shallow axis rather than looking horizontal.
        start_uv = np.array([2.1, 1.2])

        uv = [start_uv]
        for _ in range(n_steps):
            u, v = uv[-1]
            uv.append(np.array([u * fac_u, v * fac_v]))
        positions = [np.array(to_xy(u, v)) for (u, v) in uv]

        def lift(xy):
            x, y = float(xy[0]), float(xy[1])
            return np.array([x, y, bowl_z(x, y) + 0.04])

        path3d = [lift(p) for p in positions]

        dot = Dot3D(point=path3d[0], radius=0.10, color=ORANGE)
        self.play(FadeIn(dot), run_time=0.4)
        self.wait(0.3)

        trail = VGroup()
        n_seg = len(path3d) - 1
        for k in range(n_seg):
            self.play(dot.animate.move_to(path3d[k + 1]), run_time=0.6)
            seg = Line(path3d[k], path3d[k + 1], color=ORANGE, stroke_width=5)
            seg.set_opacity(0.30 + 0.10 * k)  # earlier segments fainter
            trail.add(seg)
            self.add(seg)
            self.wait(0.1)
        self.wait(0.4)

        # ------------------------------------------------------------------
        # Stage 3: tilt the camera overhead (gentle tilt, not flat). The
        # surface fades and the tilted contour rings fade in, so the bowl
        # flattens onto its own x / y axes. The descent path flattens with it:
        # the on-surface trail drops onto the base plane and stays, so the
        # curved route is still visible on the contour map at the end.
        # ------------------------------------------------------------------
        flat_pts = [np.array([float(p[0]), float(p[1]), 0.02]) for p in positions]
        flat_path = VGroup(
            *[
                Line(flat_pts[k], flat_pts[k + 1], color=ORANGE, stroke_width=3.5)
                for k in range(n_seg)
            ]
        )
        flat_dots = VGroup(
            *[Dot3D(point=p, radius=0.06, color=ORANGE) for p in flat_pts]
        )

        self.move_camera(
            phi=PHI_FINAL,
            theta=THETA_FINAL,
            run_time=2.6,
            zoom=0.78,
            added_anims=[
                surface.animate.set_opacity(0.06),
                FadeOut(dot),
                ReplacementTransform(trail, flat_path),
                FadeIn(flat_dots),
                FadeIn(contours),
            ],
        )
        self.remove(surface, dot)
        self.wait(0.5)

        # ------------------------------------------------------------------
        # Stage 4: faint TEASER arrows along the DIAGONAL principal directions
        # (NOT along x / y), hinting that the steep and shallow directions are
        # not the coordinate axes. Orange along the short axis (steep); blue
        # along the long axis (shallow). Tags stay upright. Kept faint so the
        # eigendirection reveal still belongs to the keystone slide.
        # ------------------------------------------------------------------
        u_dir = np.array([_C, _S, 0.0])  # shallow / long principal axis
        v_dir = np.array([-_S, _C, 0.0])  # steep / short principal axis
        base = np.array([0.0, 0.0, 0.05])

        shallow = Arrow(
            start=base,
            end=base + 2.5 * u_dir,
            color=BLUE,
            buff=0.0,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.13,
        )
        steep = Arrow(
            start=base,
            end=base + 1.7 * v_dir,
            color=ORANGE,
            buff=0.0,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.18,
        )
        shallow.set_opacity(0.55)
        steep.set_opacity(0.55)

        shallow_tag = Text("shallow: drops slowly", color=FG, font_size=24)
        shallow_tag.move_to(2.9 * u_dir + np.array([0.0, -0.45, 0.05]))
        steep_tag = Text("steep: drops fast", color=FG, font_size=24)
        steep_tag.move_to(2.05 * v_dir + np.array([0.0, 0.35, 0.05]))

        shallow_tag.set_opacity(0.0)
        steep_tag.set_opacity(0.0)
        self.add_fixed_orientation_mobjects(shallow_tag, steep_tag)

        self.play(FadeIn(shallow), FadeIn(steep), run_time=0.9)
        self.play(
            shallow_tag.animate.set_opacity(1.0),
            steep_tag.animate.set_opacity(1.0),
            run_time=0.7,
        )

        # ------------------------------------------------------------------
        # Stage 5: hold the composed tilted frame (axes + tilted contours +
        # curved descent path + faint diagonal teaser arrows) so a skipped /
        # looped video still ends on the correct still.
        # ------------------------------------------------------------------
        self.wait(3.0)
