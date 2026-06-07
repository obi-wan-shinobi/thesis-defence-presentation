"""Shared visual theme for thesis-defence Manim scenes.

Mirrors slides/style.sty so animations and slides read as one piece. Import
the palette constants directly, and/or subclass ThemedScene to get the deck's
light background applied automatically.

LIGHT-BACKGROUND GOTCHA: Manim's default foreground is white, which is
invisible on this background. ThemedScene only sets the *canvas* colour —
every mobject you add (axes, ticks, labels, Text/MathTex, curves, ...) still
needs an explicit dark colour from this palette, or it will vanish.
"""

from manim import TAU, Axes, Scene, color_gradient

# ----------------------------------------------------------------------
# Palette — must stay in sync with the \definecolor entries in
# slides/style.sty (bg, text, muted, tudcyan, blue, purple, orange).
# ----------------------------------------------------------------------
BG = "#F8FAFC"
FG = "#111827"
AXIS_GRAY = "#9CA3AF"
CYAN = "#00A6D6"
BLUE = "#2563EB"
PURPLE = "#7C3AED"
ORANGE = "#D97706"
MUTED = "#6B7280"


class ThemedScene(Scene):
    """Base scene that paints the deck's light background canvas.

    Subclass this instead of Scene so every animation in the deck starts
    from the same look; you are still responsible for colouring every
    mobject you add (see the gotcha above).
    """

    def setup(self):
        super().setup()
        self.camera.background_color = BG


def frequency_gradient(n, low=BLUE, mid=PURPLE, high=ORANGE):
    """n colours that read as "frequency rising" (cool blue -> warm orange).

    Use this whenever a scene needs to colour a sequence of modes/components
    by frequency, so the convention matches across animations.
    """
    return color_gradient([low, mid, high], n)


def make_glyph(fn, color, x_range=(0, TAU), x_length=2.0, y_length=0.46, stroke_width=2.75):
    """A small, axis-free curve glyph — e.g. for a stack of waveform thumbnails.

    Built via a throwaway Axes used purely as a coordinate map: the axes are
    never added to the scene, so only the plotted curve is visible. The
    caller is free to .move_to() / .scale() the returned curve into place.
    """
    helper_axes = Axes(
        x_range=[x_range[0], x_range[1]],
        y_range=[-1.05, 1.05],
        x_length=x_length,
        y_length=y_length,
        tips=False,
    )
    return helper_axes.plot(fn, color=color, stroke_width=stroke_width)
