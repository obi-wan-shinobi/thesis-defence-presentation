"""Opener animation: build the thesis target function (eq. 5.23) from its
frequency components, coarse to fine, as a running partial sum.

Render:
    manim -qh manim/src/target_construction.py TargetConstruction
    manim -qh -s manim/src/target_construction.py TargetConstruction
"""

from manim import *
import numpy as np

from theme import BG, FG, AXIS_GRAY, CYAN, MUTED, ThemedScene, frequency_gradient, make_glyph

# ----------------------------------------------------------------------
# The target function f(phi), eq. 5.23, as 7 frequency components k = 0..6
# (this is THIS scene's content — the shared palette/helpers live in theme.py)
# ----------------------------------------------------------------------
COMPONENT_FNS = [
    lambda p: 1.0,
    lambda p: 0.9 * np.cos(p),
    lambda p: -0.8 * np.sin(2 * p),
    lambda p: 0.7 * np.cos(3 * p),
    lambda p: 0.6 * np.cos(4 * p),
    lambda p: -0.5 * np.sin(5 * p),
    lambda p: -0.4 * np.sin(6 * p),
]

COMPONENT_LABELS = [
    r"1.0",
    r"0.9\cos\varphi",
    r"-0.8\sin 2\varphi",
    r"0.7\cos 3\varphi",
    r"0.6\cos 4\varphi",
    r"-0.5\sin 5\varphi",
    r"-0.4\sin 6\varphi",
]

# k = 0 (the constant) is muted gray; k = 1..6 ramp blue -> purple -> orange
# via the shared "frequency rising" gradient convention.
COMPONENT_COLORS = [MUTED] + frequency_gradient(6)


def partial_sum(up_to_k):
    """f restricted to components 0..up_to_k (inclusive), coarse to fine."""
    fns = COMPONENT_FNS[: up_to_k + 1]

    def f(phi):
        total = 0.0
        for fn in fns:
            total += fn(phi)
        return total

    return f


# Fixed y-range for the running partial-sum plot: sample the FINAL target
# over [0, 2*pi], take min/max, pad ~15%. Every partial sum k = 0..6 stays
# inside this range (adding higher modes only adds detail, not overall
# scale), so the axes never need to rescale mid-animation.
_phi_grid = np.linspace(0, TAU, 2000)
_final_vals = partial_sum(6)(_phi_grid)
_raw_min, _raw_max = float(_final_vals.min()), float(_final_vals.max())
_pad = 0.15 * (_raw_max - _raw_min)
# Round outward to whole numbers for clean, unlabelled tick marks; this is
# slightly more generous than the literal 15% but keeps the axis tidy, and
# every partial sum k = 0..6 still sits comfortably inside it.
Y_MIN = float(np.floor(_raw_min - _pad))
Y_MAX = float(np.ceil(_raw_max + _pad))


class TargetConstruction(ThemedScene):
    def construct(self):
        # ------------------------------------------------------------
        # LEFT: vertical stack of the 7 components (top = k=0, bottom = k=6)
        # ------------------------------------------------------------
        rows = VGroup()
        for k in range(7):
            glyph = make_glyph(COMPONENT_FNS[k], COMPONENT_COLORS[k])
            label = MathTex(COMPONENT_LABELS[k], color=FG).scale(0.56)
            row = VGroup(glyph, label).arrange(RIGHT, buff=0.32)
            rows.add(row)
        rows.arrange(DOWN, buff=0.22, aligned_edge=LEFT)

        # Fit the whole stack into the left ~35% of the 16:9 frame.
        fit_scale = min(4.7 / rows.width, 7.0 / rows.height)
        rows.scale(fit_scale)
        rows.to_edge(LEFT, buff=0.55)

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

        x_ticks = VGroup(*[
            MathTex(tex, color=FG).scale(0.52).next_to(axes.c2p(val, 0), DOWN, buff=0.18)
            for val, tex in [(0, "0"), (PI, r"\pi"), (TAU, r"2\pi")]
        ])

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
            LaggedStart(*[FadeIn(row, shift=RIGHT * 0.25) for row in rows], lag_ratio=0.16),
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
        # 2-3. Step through k = 0..6: highlight -> fly into axes ->
        #      morph the partial-sum curve -> mark as consumed.
        # ------------------------------------------------------------
        partial_graph = None
        for k in range(7):
            row = rows[k]
            color = COMPONENT_COLORS[k]

            highlight = SurroundingRectangle(
                row, color=color, buff=0.12, stroke_width=2.5, corner_radius=0.06
            )
            self.play(Create(highlight), run_time=0.45)

            # Slide/fade the component glyph "into" the right-hand axes.
            flying = row[0].copy()
            self.play(
                flying.animate.scale(1.7).move_to(axes.get_center()).set_opacity(0.0),
                run_time=0.7,
            )
            self.remove(flying)

            new_graph = axes.plot(partial_sum(k), color=CYAN, stroke_width=4.5)
            counter_anim = running_indicator[1].animate.set_value(k)
            if partial_graph is None:
                self.play(Create(new_graph), counter_anim, run_time=1.15)
            else:
                self.play(
                    ReplacementTransform(partial_graph, new_graph),
                    counter_anim,
                    run_time=1.15,
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
