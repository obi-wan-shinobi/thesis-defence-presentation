"""
Standalone Manim scenes for spectral-bias training dynamics.

Renders a two-panel animation:
  - left: fitted function vs target over time,
  - right: per-mode residual magnitude decay curves over steps (log y-scale).

Usage example:
  python -m manim scripts/spectral_bias_manim_scene.py SpectralBiasFitAndModes -qh
  python -m manim scripts/spectral_bias_manim_scene.py SpectralBiasSlowSingleLayer -qh
  python -m manim scripts/spectral_bias_manim_scene.py SpectralBiasModeLegend -s -qh

Optional environment variables:
  NTK_SB_RUN_DIR        Absolute/relative run dir (overrides auto-latest).
  NTK_SB_BASE_RESULTS   Results root (default: <repo>/data).
  NTK_SB_EXP_NAME       Experiment name (default: ntk_spectral_bias_architecture_comparison).
  NTK_SB_MODEL          Model name inside run (default: first configured model).
  NTK_SB_SEED_INDEX     Seed index (default: 0).
  NTK_SB_FRAME_STRIDE   Use every k-th snapshot (default: 1).
  NTK_SB_GAMMA_STRIDE   Downsample gamma by stride for plotting (default: 1).

Edit the scene controls below to adjust animation timing.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from theme import BG

from manim import (
    BLACK,
    BLUE_D,
    DOWN,
    GREEN_D,
    LEFT,
    ORANGE,
    PURPLE_D,
    RED_D,
    RIGHT,
    UP,
    WHITE,
    Axes,
    DecimalNumber,
    Line,
    MathTex,
    Scene,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    config,
    linear,
)

config.background_color = WHITE


# ---------------------------------------------------------------------------
# Scene controls
# ---------------------------------------------------------------------------

DEFAULT_RUNTIME_SEC = 20.0

# Slow-intro scene timing. Increase SLOW_END_STEP to let the training animation
# run further before stopping. The schedule keeps the early residual panel
# zoomed in, then widens it once the counter passes SLOW_ZOOM_OUT_STEP.
SLOW_RUNTIME_SEC = 28.0
SLOW_END_STEP = 10000.0
SLOW_INITIAL_X_WINDOW = 1200.0
SLOW_ZOOM_OUT_STEP = 1200.0
SLOW_ZOOM_CATCHUP_STEP = 1500.0
SLOW_ZOOM_CATCHUP_X_WINDOW = 3200.0
SLOW_FINAL_X_WINDOW = 12000.0
SLOW_FINAL_X_WINDOW_PAD = 1.15
SLOW_FINAL_HOLD_SEC = 1.0

# Relative time spent on 0-100, 100-600, 600-zoom, zoom catch-up, and catch-up-end.
# Smaller early weights make the start less sluggish; larger late weights slow
# down the residual decay once the panel has zoomed out. During the catch-up
# segment, the x-window widens faster than the residual point advances.
SLOW_SEGMENT_WEIGHTS = (0.07, 0.11, 0.17, 0.08, 0.57)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _latest_run_dir(base_results: Path, exp_name: str) -> Path:
    exp_root = base_results / exp_name
    latest = exp_root / "latest"
    if latest.exists():
        return latest.resolve()

    runs = sorted([p for p in exp_root.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run directories found under {exp_root}")
    return runs[-1]


def _read_npz(path: Path) -> dict:
    return dict(np.load(path, allow_pickle=True))


def _resolve_model_name(manifest: dict, run_dir: Path) -> str:
    env_model = os.getenv("NTK_SB_MODEL")
    if env_model:
        if env_model not in manifest["runs"]:
            raise KeyError(
                f"NTK_SB_MODEL='{env_model}' not found in run models: "
                f"{sorted(manifest['runs'].keys())}"
            )
        return env_model

    network_cfg_file = run_dir / manifest.get("network_configs", "")
    if network_cfg_file.exists():
        cfg = json.loads(network_cfg_file.read_text())
        names = [
            str(m["name"])
            for m in cfg.get("models", [])
            if str(m["name"]) in manifest["runs"]
        ]
        if names:
            return names[0]

    return sorted(manifest["runs"].keys())[0]


def _prepare_scene_data() -> dict:
    repo_root = Path(__file__).resolve().parents[1]

    base_results = Path(
        os.getenv("NTK_SB_BASE_RESULTS", str(repo_root / "data"))
    ).expanduser()
    exp_name = os.getenv("NTK_SB_EXP_NAME", "ntk_spectral_bias_architecture_comparison")

    run_dir_env = os.getenv("NTK_SB_RUN_DIR")
    run_dir = (
        Path(run_dir_env).expanduser().resolve()
        if run_dir_env
        else _latest_run_dir(base_results, exp_name)
    )

    manifest = json.loads((run_dir / "manifest.json").read_text())
    probe = _read_npz(run_dir / manifest["probe_geometry"])
    target = _read_npz(run_dir / manifest["target_task"])
    mode_bank = _read_npz(run_dir / manifest["mode_bank"])

    model_name = _resolve_model_name(manifest, run_dir)
    run = _read_npz(run_dir / manifest["runs"][model_name])

    seed_index = int(os.getenv("NTK_SB_SEED_INDEX", "0"))
    seeds = np.asarray(run["seeds"], dtype=int)
    if seed_index < 0 or seed_index >= len(seeds):
        raise IndexError(
            f"NTK_SB_SEED_INDEX={seed_index} out of range for seeds={seeds.tolist()}"
        )

    gamma_key = "gamma_probe" if "gamma_probe" in probe else "gamma_eval"
    target_key = "y_probe_true" if "y_probe_true" in target else "y_eval_true"

    gamma = np.asarray(probe[gamma_key], dtype=float)
    y_true = np.asarray(target[target_key], dtype=float)

    pred = np.asarray(run["pred_eval"][seed_index], dtype=float)
    mode_abs = np.asarray(run["target_mode_abs_eval"][seed_index], dtype=float)
    steps = np.asarray(run["snapshot_steps"], dtype=float)
    mode_names = [str(x) for x in mode_bank["tracked_mode_names"]]

    frame_stride = max(1, int(os.getenv("NTK_SB_FRAME_STRIDE", "1")))
    frame_indices = list(range(0, len(steps), frame_stride))
    if frame_indices[-1] != len(steps) - 1:
        frame_indices.append(len(steps) - 1)

    gamma_stride = max(1, int(os.getenv("NTK_SB_GAMMA_STRIDE", "1")))
    gamma = gamma[::gamma_stride]
    y_true = y_true[::gamma_stride]
    pred = pred[:, ::gamma_stride]

    pred = pred[frame_indices]
    mode_abs = mode_abs[frame_indices]
    steps = steps[frame_indices]

    return {
        "run_dir": str(run_dir),
        "model_name": model_name,
        "seed": int(seeds[seed_index]),
        "probe_domain": str(manifest.get("meta", {}).get("probe_domain", "eval")),
        "gamma": gamma,
        "target": y_true,
        "pred": pred,
        "mode_abs": mode_abs,
        "steps": steps,
        "mode_names": mode_names,
        "runtime_sec": DEFAULT_RUNTIME_SEC,
    }


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------


def _interp_rows(values: np.ndarray, t: float) -> np.ndarray:
    n = int(values.shape[0])
    if n == 1:
        return values[0]
    t = float(np.clip(t, 0.0, n - 1))
    i0 = int(np.floor(t))
    i1 = min(i0 + 1, n - 1)
    a = t - i0
    return (1.0 - a) * values[i0] + a * values[i1]


# ---------------------------------------------------------------------------
# Log-scale helpers for the right panel
# ---------------------------------------------------------------------------
# We manually transform to log10 space and shift so y starts at 0,
# keeping the right-panel x-axis at the bottom.

LOG_EPS = 1e-9  # safe floor to avoid log(0)


def _shifted_log10(x: np.ndarray) -> tuple[np.ndarray, float]:
    log_x = np.log10(np.maximum(x, LOG_EPS))
    log_floor = float(np.min(log_x))
    return log_x - log_floor, log_floor


def _mode_name_to_latex(name: str, fallback_index: int) -> str:
    mode_match = re.search(r"\b(cos|sin)[_\-\s]*(\d+)?\b", name.lower())
    mode = mode_match.group(1) if mode_match else "cos"
    k = (
        int(mode_match.group(2))
        if mode_match and mode_match.group(2)
        else fallback_index
    )
    coefficient = "" if k == 1 else f" {k}"
    return rf"\{mode}{coefficient}\gamma"


def _fit_y_range(target: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    all_fit = np.concatenate([target, pred.ravel()])
    y_min_raw = float(np.min(all_fit))
    y_max_raw = float(np.max(all_fit))
    y_span = max(y_max_raw - y_min_raw, 1e-9)
    pad = 0.08 * y_span
    y_bot = y_min_raw - pad
    y_top = y_max_raw + pad
    y_step = max(0.2, (y_top - y_bot) / 6.0)
    return y_bot, y_top, y_step


def _step_at_frame(steps: np.ndarray, frame_value: float) -> float:
    n = int(steps.shape[0])
    if n == 1:
        return float(steps[0])
    frame_value = float(np.clip(frame_value, 0.0, n - 1))
    i0 = int(np.floor(frame_value))
    i1 = min(i0 + 1, n - 1)
    a = frame_value - i0
    return float((1.0 - a) * steps[i0] + a * steps[i1])


def _frame_for_step(steps: np.ndarray, step: float) -> float:
    if len(steps) <= 1:
        return 0.0
    return float(np.interp(step, steps, np.arange(len(steps), dtype=float)))


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


class SpectralBiasFitAndModes(Scene):
    def construct(self):
        data = _prepare_scene_data()

        gamma = data["gamma"]
        target = data["target"]
        pred = data["pred"]
        mode_abs = data["mode_abs"]
        steps = data["steps"]

        # ------------------------------------------------------------------
        # Left panel: auto-range y to include negative values
        # ------------------------------------------------------------------
        all_fit = np.concatenate([target, pred.ravel()])
        y_min_raw = float(np.min(all_fit))
        y_max_raw = float(np.max(all_fit))
        y_span = max(y_max_raw - y_min_raw, 1e-9)
        pad = 0.08 * y_span
        y_bot = y_min_raw - pad
        y_top = y_max_raw + pad
        y_step = max(0.2, (y_top - y_bot) / 6.0)

        left_axes = Axes(
            x_range=[0.0, 2.0 * np.pi, np.pi / 2],
            y_range=[y_bot, y_top, y_step],
            x_length=5.95,
            y_length=5.15,
            axis_config={"include_numbers": False, "color": BLACK},
        ).to_edge(LEFT, buff=0.15)

        # ------------------------------------------------------------------
        # Right panel: log-scale y (manual transform)
        # ------------------------------------------------------------------
        mode_vals = np.clip(mode_abs, 0.0, None)
        log_mode, _ = _shifted_log10(mode_vals)  # shape: (frames, modes)

        log_max = float(np.max(log_mode))
        log_top = log_max + 0.08 * max(log_max, 1.0)
        if log_top <= 0.0:
            log_top = 1.0
        log_step = max(0.2, log_top / 6.0)

        right_axes = Axes(
            x_range=[
                float(steps[0]),
                float(steps[-1]),
                max(1.0, (steps[-1] - steps[0]) / 5),
            ],
            y_range=[0.0, log_top, log_step],
            x_length=5.95,
            y_length=5.15,
            axis_config={"include_numbers": False, "color": BLACK},
        ).to_edge(RIGHT, buff=0.15)

        # ------------------------------------------------------------------
        # Axis labels
        # ------------------------------------------------------------------
        left_y_label = MathTex(r"f(\gamma)", font_size=34, color=BLACK).rotate(
            np.pi / 2
        )
        left_y_label.move_to(left_axes.c2p(0.75, y_top - 0.18 * (y_top - y_bot)))
        left_x_label = MathTex(r"\gamma", font_size=36, color=BLACK).next_to(
            left_axes, DOWN, buff=0.2
        )
        right_y_label = (
            MathTex(
                r"\langle r_t(\gamma),\,\phi_k(\gamma)\rangle",
                font_size=22,
                color=BLACK,
            )
            .rotate(np.pi / 2)
            .next_to(right_axes, LEFT, buff=0.14)
        )
        right_x_label = MathTex(r"t", font_size=36, color=BLACK).next_to(
            right_axes, DOWN, buff=0.2
        )

        # ------------------------------------------------------------------
        # Static target curve (left panel)
        # ------------------------------------------------------------------
        target_curve = left_axes.plot_line_graph(
            x_values=gamma.tolist(),
            y_values=target.tolist(),
            add_vertex_dots=False,
            line_color=BLACK,
            stroke_width=4,
        )

        # ------------------------------------------------------------------
        # Animated fit curve (left panel)
        # ------------------------------------------------------------------
        frame = ValueTracker(0)

        def fit_curve_fn():
            y = _interp_rows(pred, frame.get_value())
            return left_axes.plot_line_graph(
                x_values=gamma.tolist(),
                y_values=y.tolist(),
                add_vertex_dots=False,
                line_color=BLUE_D,
                stroke_width=5,
            )

        fit_curve = always_redraw(fit_curve_fn)

        # ------------------------------------------------------------------
        # Animated residual-mode curves (right panel, log-space y)
        # ------------------------------------------------------------------
        mode_palette = [BLUE_D, GREEN_D, RED_D, PURPLE_D, ORANGE]
        mode_curves = VGroup()

        for m in range(log_mode.shape[1]):
            color = mode_palette[m % len(mode_palette)]

            def mode_curve_fn(m_idx=m, c=color):
                t = float(np.clip(frame.get_value(), 0.0, len(steps) - 1))
                i0 = int(np.floor(t))
                i1 = min(i0 + 1, len(steps) - 1)
                a = t - i0

                step_t = (1.0 - a) * steps[i0] + a * steps[i1]
                log_t = (1.0 - a) * log_mode[i0, m_idx] + a * log_mode[i1, m_idx]

                pts = [
                    right_axes.c2p(float(steps[j]), float(log_mode[j, m_idx]))
                    for j in range(i0 + 1)
                ]
                pts.append(right_axes.c2p(float(step_t), float(log_t)))
                if len(pts) == 1:
                    pts.append(pts[0])

                curve = VMobject(color=c, stroke_width=4)
                curve.set_points_as_corners(pts)
                return curve

            mode_curves.add(always_redraw(mode_curve_fn))

        # ------------------------------------------------------------------
        # Compose scene
        # ------------------------------------------------------------------
        self.add(
            left_axes,
            right_axes,
            left_y_label,
            left_x_label,
            right_y_label,
            right_x_label,
            target_curve,
            fit_curve,
            mode_curves,
        )
        self.play(
            frame.animate.set_value(len(steps) - 1),
            run_time=float(data["runtime_sec"]),
            rate_func=linear,
        )
        self.wait(0.3)


class SpectralBiasSlowSingleLayer(Scene):
    """Same visual language as SpectralBiasFitAndModes, with slow early time.

    The saved experiment has snapshots every 100 optimization steps. This scene
    deliberately spends a large fraction of its runtime between the first few
    snapshots so the low-frequency onset is visible in the talk, without
    touching the existing architecture-comparison MP4s.
    """

    def construct(self):
        self.camera.background_color = BG
        data = _prepare_scene_data()

        gamma = data["gamma"]
        target = data["target"]
        pred = data["pred"]
        mode_abs = data["mode_abs"]
        steps = data["steps"]
        mode_names = data["mode_names"]

        y_bot, y_top, y_step = _fit_y_range(target, pred)

        left_axes = Axes(
            x_range=[0.0, 2.0 * np.pi, np.pi / 2],
            y_range=[y_bot, y_top, y_step],
            x_length=5.95,
            y_length=5.15,
            axis_config={"include_numbers": False, "color": BLACK},
        ).to_edge(LEFT, buff=0.15)

        mode_vals = np.clip(mode_abs, 0.0, None)
        log_mode, _ = _shifted_log10(mode_vals)

        log_max = float(np.max(log_mode))
        log_top = log_max + 0.08 * max(log_max, 1.0)
        if log_top <= 0.0:
            log_top = 1.0
        log_step = max(0.2, log_top / 6.0)

        frame = ValueTracker(0)
        x_window = ValueTracker(min(float(steps[-1]), 600.0))
        x_start = float(steps[0])
        x_final = float(steps[-1])
        right_plot_box = Axes(
            x_range=[0.0, 1.0, 1.0],
            y_range=[0.0, 1.0, 1.0],
            x_length=5.95,
            y_length=5.15,
            axis_config={"include_numbers": False, "color": BLACK},
        ).to_edge(RIGHT, buff=0.15)
        plot_left = right_plot_box.c2p(0.0, 0.0)[0]
        plot_right = right_plot_box.c2p(1.0, 0.0)[0]
        plot_bottom = right_plot_box.c2p(0.0, 0.0)[1]
        plot_top = right_plot_box.c2p(0.0, 1.0)[1]

        def current_x_max() -> float:
            return max(x_start + 1.0, min(float(x_window.get_value()), x_final))

        def right_panel_point(step: float, log_value: float) -> np.ndarray:
            x_alpha = (float(step) - x_start) / (current_x_max() - x_start)
            y_alpha = float(log_value) / log_top
            x = plot_left + np.clip(x_alpha, 0.0, 1.0) * (plot_right - plot_left)
            y = plot_bottom + np.clip(y_alpha, 0.0, 1.0) * (plot_top - plot_bottom)
            return np.array([x, y, 0.0])

        def right_axes_fn():
            x_max = current_x_max()
            return Axes(
                x_range=[x_start, x_max, max(1.0, (x_max - x_start) / 5)],
                y_range=[0.0, log_top, log_step],
                x_length=5.95,
                y_length=5.15,
                axis_config={"include_numbers": False, "color": BLACK},
            ).to_edge(RIGHT, buff=0.15)

        right_axes = always_redraw(right_axes_fn)

        left_y_label = MathTex(r"f(\gamma)", font_size=34, color=BLACK).rotate(
            np.pi / 2
        )
        left_y_label.move_to(left_axes.c2p(0.75, y_top - 0.18 * (y_top - y_bot)))
        left_x_label = MathTex(r"\gamma", font_size=36, color=BLACK).next_to(
            left_axes, DOWN, buff=0.2
        )
        right_y_label = (
            MathTex(
                r"\langle r_t(\gamma),\,\phi_k(\gamma)\rangle",
                font_size=22,
                color=BLACK,
            )
            .rotate(np.pi / 2)
            .next_to(right_axes, LEFT, buff=0.14)
        )
        right_x_label = MathTex(r"t", font_size=36, color=BLACK).next_to(
            right_axes, DOWN, buff=0.2
        )

        target_curve = left_axes.plot_line_graph(
            x_values=gamma.tolist(),
            y_values=target.tolist(),
            add_vertex_dots=False,
            line_color=BLACK,
            stroke_width=4,
        )

        def fit_curve_fn():
            y = _interp_rows(pred, frame.get_value())
            return left_axes.plot_line_graph(
                x_values=gamma.tolist(),
                y_values=y.tolist(),
                add_vertex_dots=False,
                line_color=BLUE_D,
                stroke_width=5,
            )

        fit_curve = always_redraw(fit_curve_fn)

        mode_palette = [BLUE_D, GREEN_D, RED_D, PURPLE_D, ORANGE]
        mode_curves = VGroup()
        mode_labels = VGroup()

        def current_mode_path(m_idx: int) -> tuple[np.ndarray, np.ndarray]:
            t = float(np.clip(frame.get_value(), 0.0, len(steps) - 1))
            i0 = int(np.floor(t))
            i1 = min(i0 + 1, len(steps) - 1)
            a = t - i0

            step_t = float((1.0 - a) * steps[i0] + a * steps[i1])
            log_t = float((1.0 - a) * log_mode[i0, m_idx] + a * log_mode[i1, m_idx])

            step_path = np.asarray(steps[: i0 + 1], dtype=float)
            log_path = np.asarray(log_mode[: i0 + 1, m_idx], dtype=float)
            if step_t > step_path[-1] + 1e-9:
                step_path = np.append(step_path, step_t)
                log_path = np.append(log_path, log_t)
            else:
                step_path = step_path.copy()
                log_path = log_path.copy()
                step_path[-1] = step_t
                log_path[-1] = log_t

            visible_x = min(step_path[-1], current_x_max())
            keep = step_path <= visible_x + 1e-9
            visible_steps = step_path[keep]
            visible_logs = log_path[keep]

            if visible_steps[-1] < visible_x and len(step_path) > len(visible_steps):
                visible_steps = np.append(visible_steps, visible_x)
                visible_logs = np.append(
                    visible_logs,
                    float(np.interp(visible_x, step_path, log_path)),
                )

            return visible_steps, visible_logs

        def mode_label_y_offset(m_idx: int) -> float:
            tip_ys = []
            for idx in range(log_mode.shape[1]):
                visible_steps, visible_logs = current_mode_path(idx)
                tip = right_panel_point(
                    float(visible_steps[-1]),
                    float(visible_logs[-1]),
                )
                tip_ys.append(float(tip[1]))

            if len(tip_ys) == 1:
                return 0.0

            order = sorted(
                range(len(tip_ys)), key=lambda idx: tip_ys[idx], reverse=True
            )
            rank = order.index(m_idx)
            offsets = np.linspace(0.24, -0.24, len(tip_ys))
            return float(offsets[rank])

        for m in range(log_mode.shape[1]):
            color = mode_palette[m % len(mode_palette)]
            label_template = MathTex(
                _mode_name_to_latex(mode_names[m], fallback_index=m + 1),
                font_size=20,
                color=color,
            )

            def mode_curve_fn(m_idx=m, c=color):
                visible_steps, visible_logs = current_mode_path(m_idx)
                pts = [
                    right_panel_point(float(step), float(log_value))
                    for step, log_value in zip(visible_steps, visible_logs)
                ]
                if len(pts) == 1:
                    pts.append(pts[0])

                curve = VMobject(color=c, stroke_width=4)
                curve.set_points_as_corners(pts)
                return curve

            mode_curves.add(always_redraw(mode_curve_fn))

            def mode_label_fn(m_idx=m, template=label_template):
                visible_steps, visible_logs = current_mode_path(m_idx)
                tip = right_panel_point(
                    float(visible_steps[-1]), float(visible_logs[-1])
                )
                label_mob = template.copy()

                x = tip[0] + 0.1 + label_mob.width / 2
                y = tip[1] + mode_label_y_offset(m_idx)
                x = np.clip(
                    x,
                    plot_left + label_mob.width / 2,
                    plot_right - label_mob.width / 2,
                )
                y = np.clip(
                    y,
                    plot_bottom + label_mob.height / 2,
                    plot_top - label_mob.height / 2,
                )
                label_mob.move_to([x, y, 0.0])
                return label_mob

            mode_labels.add(always_redraw(mode_label_fn))

        counter_label = MathTex(r"\mathrm{step}", font_size=44, color=BLACK)
        counter_value = DecimalNumber(
            0,
            num_decimal_places=0,
            group_with_commas=False,
            font_size=44,
            color=BLACK,
        )

        def update_counter_value(mob):
            mob.set_value(_step_at_frame(steps, frame.get_value()))
            mob.next_to(counter_label, RIGHT, buff=0.16)

        counter_value.add_updater(update_counter_value)
        counter = VGroup(counter_label, counter_value).arrange(RIGHT, buff=0.16)
        counter.next_to(left_axes, UP, buff=0.12).align_to(left_axes, LEFT)

        self.add(
            left_axes,
            right_axes,
            left_y_label,
            left_x_label,
            right_y_label,
            right_x_label,
            target_curve,
            fit_curve,
            mode_curves,
            mode_labels,
            counter,
        )

        runtime = float(SLOW_RUNTIME_SEC)
        end_step = min(
            float(steps[-1]),
            max(float(steps[0]), float(SLOW_END_STEP)),
        )
        final_x_window = min(
            x_final,
            max(
                x_start + 1.0,
                float(SLOW_FINAL_X_WINDOW),
                end_step * float(SLOW_FINAL_X_WINDOW_PAD),
            ),
        )
        initial_x_window = min(
            x_final,
            max(
                x_start + 1.0,
                float(SLOW_INITIAL_X_WINDOW),
            ),
        )
        zoom_out_step = min(
            end_step,
            max(float(steps[0]), float(SLOW_ZOOM_OUT_STEP)),
        )
        zoom_catchup_step = min(
            end_step,
            max(zoom_out_step, float(SLOW_ZOOM_CATCHUP_STEP)),
        )
        zoom_catchup_x_window = min(
            x_final,
            max(
                initial_x_window,
                float(SLOW_ZOOM_CATCHUP_X_WINDOW),
            ),
        )
        schedule_specs = [
            (min(end_step, 100.0), initial_x_window, SLOW_SEGMENT_WEIGHTS[0]),
            (min(end_step, 600.0), initial_x_window, SLOW_SEGMENT_WEIGHTS[1]),
            (zoom_out_step, initial_x_window, SLOW_SEGMENT_WEIGHTS[2]),
            (
                zoom_catchup_step,
                zoom_catchup_x_window,
                SLOW_SEGMENT_WEIGHTS[3],
            ),
            (end_step, final_x_window, SLOW_SEGMENT_WEIGHTS[4]),
        ]

        schedule = []
        current_step = float(steps[0])
        for target_step, target_x_window, weight in schedule_specs:
            target_step = float(target_step)
            if target_step <= current_step + 1e-9:
                continue
            schedule.append((target_step, target_x_window, float(weight)))
            current_step = target_step

        segment_weights = np.asarray(
            [weight for _, _, weight in schedule],
            dtype=float,
        )
        segment_weights = segment_weights / float(np.sum(segment_weights))

        for (target_step, target_x_window, _), weight in zip(
            schedule, segment_weights
        ):
            target_frame = _frame_for_step(steps, target_step)
            if target_frame <= frame.get_value():
                continue
            self.play(
                frame.animate.set_value(target_frame),
                x_window.animate.set_value(target_x_window),
                run_time=runtime * weight,
                rate_func=linear,
            )
        self.wait(float(SLOW_FINAL_HOLD_SEC))


class SpectralBiasModeLegend(Scene):
    def construct(self):
        data = _prepare_scene_data()
        mode_names = data["mode_names"]

        mode_palette = [BLUE_D, GREEN_D, RED_D, PURPLE_D, ORANGE]
        legend_items = VGroup()

        for m, name in enumerate(mode_names):
            color = mode_palette[m % len(mode_palette)]
            line = Line(LEFT * 1.0, RIGHT * 1.0, color=color, stroke_width=8)
            label = MathTex(
                _mode_name_to_latex(name, fallback_index=m + 1),
                font_size=48,
                color=BLACK,
            )
            legend_items.add(VGroup(line, label).arrange(RIGHT, buff=0.35))

        legend_items.arrange(RIGHT, buff=0.5)
        max_width = config.frame_width - 0.8
        if legend_items.width > max_width:
            legend_items.scale_to_fit_width(max_width)
        legend_items.move_to([0, 0, 0])
        self.add(legend_items)
