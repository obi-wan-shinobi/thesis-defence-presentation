"""Opener animation: build a target function from a selected subset of its
frequency components (k = 0, 1, 3, 5, 6 — see FREQS below), coarse to fine,
as a running partial sum.

Render:
    manim -qh manim/src/target_construction.py TargetConstruction
    manim -qh -s manim/src/target_construction.py TargetConstruction
"""

import numpy as np
from theme import (
    AXIS_GRAY,
    BG,
    CYAN,
    FG,
    MUTED,
    ThemedScene,
    frequency_gradient,
    make_glyph,
)

from manim import *

# ----------------------------------------------------------------------
# The target function, built from 5 of its frequency components, k = 0, 1,
# 3, 5, 6 (k = 2 and k = 4 dropped). FREQS records each entry's TRUE
# frequency, since it no longer matches its position in these lists — the
# running indicator must display FREQS[i], not the loop position i.
# (This is THIS scene's content — the shared palette/helpers live in theme.py.)
# ----------------------------------------------------------------------
FREQS = [0, 1, 3, 5, 6]

COMPONENT_FNS = [
    lambda p: 1.0,
    lambda p: 0.9 * np.cos(p),
    lambda p: 0.7 * np.cos(3 * p),
    lambda p: -0.5 * np.sin(5 * p),
    lambda p: -0.4 * np.sin(6 * p),
]

COMPONENT_LABELS = [
    r"1.0",
    r"0.9\cos\varphi",
    r"0.7\cos 3\varphi",
    r"-0.5\sin 5\varphi",
    r"-0.4\sin 6\varphi",
]

# k = 0 (the constant) is muted gray; the remaining modes ramp blue -> purple
# -> orange via the shared "frequency rising" gradient convention.
COMPONENT_COLORS = [MUTED] + frequency_gradient(len(COMPONENT_FNS) - 1)


def partial_sum(up_to_index):
    """f restricted to components 0..up_to_index (inclusive), coarse to fine."""
    fns = COMPONENT_FNS[: up_to_index + 1]

    def f(phi):
        total = 0.0
        for fn in fns:
            total += fn(phi)
        return total

    return f


# Fixed y-range for the running partial-sum plot: sample the FINAL target
# over [0, 2*pi], take min/max, pad ~15%. Every partial sum stays inside
# this range (adding higher modes only adds detail, not overall scale), so
# the axes never need to rescale mid-animation.
_phi_grid = np.linspace(0, TAU, 2000)
_final_vals = partial_sum(len(COMPONENT_FNS) - 1)(_phi_grid)
_raw_min, _raw_max = float(_final_vals.min()), float(_final_vals.max())
_pad = 0.15 * (_raw_max - _raw_min)
# Round outward to whole numbers for clean, unlabelled tick marks; this is
# slightly more generous than the literal 15% but keeps the axis tidy, and
# every partial sum still sits comfortably inside it.
Y_MIN = float(np.floor(_raw_min - _pad))
Y_MAX = float(np.ceil(_raw_max + _pad))


class TargetConstruction(ThemedScene):
    def construct(self):
        # ------------------------------------------------------------
        # LEFT: vertical stack of the 5 components (top = lowest freq,
        # bottom = highest freq)
        # ------------------------------------------------------------
        rows = VGroup()
        for i in range(len(COMPONENT_FNS)):
            glyph = make_glyph(COMPONENT_FNS[i], COMPONENT_COLORS[i])
            label = MathTex(COMPONENT_LABELS[i], color=FG).scale(0.56)
            row = VGroup(glyph, label).arrange(RIGHT, buff=0.32)
            rows.add(row)
        rows.arrange(DOWN, buff=0.4, aligned_edge=LEFT)

        # Fit the whole stack into the left ~35% of the 16:9 frame, capped so
        # fewer rows don't blow the glyphs up too large, then centre it there
        # (its right edge stays well clear of the axes' left edge at x=-2.13).
        fit_scale = min(4.7 / rows.width, 6.4 / rows.height, 1.35)
        rows.scale(fit_scale)
        rows.move_to(LEFT * 4.6)

        # ------------------------------------------------------------
        # RIGHT: large axes for the running partial sum
        # ------------------------------------------------------------
        axes = Axes(
            x_range=[0, TAU, PI / 2],
            y_range=[Y_MIN, Y_MAX, 1],
            x_length=7.4,
            y_length=5.5,
            tips=False,
            axis_config={"color": AXIS_GRAY, "stroke_width": 2},
        )
        axes.move_to(RIGHT * 2.75 + DOWN * 0.2)

        x_label = MathTex(r"\varphi", color=FG).scale(0.7)
        x_label.next_to(axes.x_axis.get_right(), UR, buff=0.12)
        y_label = MathTex(r"f(\varphi)", color=FG).scale(0.62)
        y_label.next_to(axes.y_axis.get_top(), UP, buff=0.12)

        x_ticks = VGroup(
            *[
                MathTex(tex, color=FG)
                .scale(0.52)
                .next_to(axes.c2p(val, 0), DOWN, buff=0.18)
                for val, tex in [(0, "0"), (PI, r"\pi"), (TAU, r"2\pi")]
            ]
        )

        axes_group = VGroup(axes, x_label, y_label, x_ticks)

        running_indicator = VGroup(
            MathTex(r"\text{up to } k = ", color=MUTED).scale(0.6),
            Integer(0, color=CYAN).scale(0.8),
        ).arrange(RIGHT, buff=0.15, aligned_edge=DOWN)
        running_indicator.next_to(axes, UP, buff=0.38).align_to(axes, RIGHT)

        # ------------------------------------------------------------
        # 1. Reveal the left stack (quick stagger)
        # ------------------------------------------------------------
        self.play(
            LaggedStart(
                *[FadeIn(row, shift=RIGHT * 0.25) for row in rows], lag_ratio=0.16
            ),
            run_time=2.4,
        )
        self.wait(0.3)

        # Bring in the right-hand axes and the running indicator.
        self.play(
            Create(axes),
            FadeIn(x_label, y_label, x_ticks),
            FadeIn(running_indicator),
            run_time=1.4,
        )
        self.wait(0.4)

        # ------------------------------------------------------------
        # 2-3. Step through the 5 selected components: highlight -> bring
        #      into the axes -> update the partial-sum curve -> mark as
        #      consumed. The running indicator shows the TRUE frequency
        #      FREQS[i], which is not the same as the loop position i.
        # ------------------------------------------------------------
        partial_graph = None
        for i in range(len(COMPONENT_FNS)):
            row = rows[i]
            color = COMPONENT_COLORS[i]

            highlight = SurroundingRectangle(
                row, color=color, buff=0.12, stroke_width=2.5, corner_radius=0.06
            )
            self.play(Create(highlight), run_time=0.45)

            new_graph = axes.plot(partial_sum(i), color=CYAN, stroke_width=4.5)
            counter_anim = running_indicator[1].animate.set_value(FREQS[i])

            if i == 0:
                # First mode: morph the small flat-line glyph directly into
                # the constant partial-sum line — it visibly grows/slides
                # from the stack into the axes, rather than vanishing and
                # being redrawn from scratch.
                flying = row[0].copy()
                self.play(
                    ReplacementTransform(flying, new_graph),
                    counter_anim,
                    run_time=1.0,
                )
            else:
                # Slide/fade the component glyph "into" the right-hand axes,
                # then morph the previous partial sum into the new one.
                flying = row[0].copy()
                self.play(
                    flying.animate.scale(1.7)
                    .move_to(axes.get_center())
                    .set_opacity(0.0),
                    run_time=0.7,
                )
                self.remove(flying)

                self.play(
                    ReplacementTransform(partial_graph, new_graph),
                    counter_anim,
                    run_time=0.7,
                )
            partial_graph = new_graph

            # Mark this component as consumed: dim it on the left.
            # (Dim the glyph's *stroke* only — ParametricFunction curves are
            # closed paths with fill_opacity=0, and a blanket set_opacity
            # would reveal an unwanted fill under the wave.)
            glyph, label = row
            self.play(
                FadeOut(highlight),
                glyph.animate.set_stroke(opacity=0.32),
                label.animate.set_opacity(0.32),
                run_time=0.45,
            )

        # ------------------------------------------------------------
        # 4. Hold the finished target — this frame becomes the poster.
        # ------------------------------------------------------------
        self.wait(4.0)
