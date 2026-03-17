#!/usr/bin/env python3
"""Per-frame statistics visualizer for stella_vslam msgpack map files.

Reads the "frames" section of a stella_vslam msgpack map file (saved with
save_frames: true) and plots the following metrics as time-series graphs:

  1. num_tracked_landmarks
  2. landmark_reproj_error_px        (mean & median)
  3. landmark_parallax_deg           (mean & median)
  4. landmark_direction_variance     (1 − ‖mean unit vec‖)
  5. all_feature_direction_variance  (1 − ‖mean unit vec‖)
  6. landmark_feature_response       (mean & median)
  7. all_feature_response            (mean & median)

Usage:
    # Save to PNG
    python plot_frame_statistics.py map.msg --output-dir ./plots

    # Show interactively
    python plot_frame_statistics.py map.msg
"""

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize per-frame statistics from a stella_vslam msgpack map file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "msgpack",
        type=str,
        help="Path to the .msg / .msgpack map file (saved with save_frames: true).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save the plot as PNG. If omitted, display interactively.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI resolution for the saved PNG.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[14.0, 22.0],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_frames(msgpack_path: str) -> dict:
    """Decode the msgpack file and return the 'frames' dict."""
    import msgpack as _msgpack

    with open(msgpack_path, "rb") as fh:
        data = _msgpack.unpackb(fh.read(), raw=False)

    if "frames" not in data:
        raise KeyError(
            "The msgpack file does not contain a 'frames' key.\n"
            "Re-run stella_vslam with  save_frames: true  in the map I/O config."
        )

    return data["frames"]


def _safe_float(value) -> float:
    """Convert a value to float, mapping None/null to NaN."""
    return float("nan") if value is None else float(value)


def _summary_pair(obj) -> tuple:
    """Extract (mean, median) from a summary_statistics JSON object."""
    if obj is None:
        return float("nan"), float("nan")
    return _safe_float(obj.get("mean")), _safe_float(obj.get("median"))


def build_series(frames: dict) -> dict:
    """Sort frames by ID and extract per-field numpy arrays."""
    import numpy as np

    sorted_items = sorted(frames.items(), key=lambda kv: int(kv[0]))

    ids = []
    num_tracked          = []
    reproj_mean          = []
    reproj_median        = []
    parallax_mean        = []
    parallax_median      = []
    lm_dir_var           = []
    all_dir_var          = []
    lm_resp_mean         = []
    lm_resp_median       = []
    all_resp_mean        = []
    all_resp_median      = []

    for fid_str, fd in sorted_items:
        ids.append(int(fid_str))

        num_tracked.append(_safe_float(fd.get("num_tracked_landmarks")))

        m, med = _summary_pair(fd.get("landmark_reproj_error_px"))
        reproj_mean.append(m);   reproj_median.append(med)

        m, med = _summary_pair(fd.get("landmark_parallax_deg"))
        parallax_mean.append(m); parallax_median.append(med)

        lm_dir_var.append(_safe_float(fd.get("landmark_direction_variance")))
        all_dir_var.append(_safe_float(fd.get("all_feature_direction_variance")))

        m, med = _summary_pair(fd.get("landmark_feature_response"))
        lm_resp_mean.append(m);  lm_resp_median.append(med)

        m, med = _summary_pair(fd.get("all_feature_response"))
        all_resp_mean.append(m); all_resp_median.append(med)

    a = np.asarray  # shorthand

    return {
        "frame_ids":                        a(ids,             dtype=float),
        "num_tracked_landmarks":            a(num_tracked,     dtype=float),
        "landmark_reproj_error_px_mean":    a(reproj_mean,     dtype=float),
        "landmark_reproj_error_px_median":  a(reproj_median,   dtype=float),
        "landmark_parallax_deg_mean":       a(parallax_mean,   dtype=float),
        "landmark_parallax_deg_median":     a(parallax_median, dtype=float),
        "landmark_direction_variance":      a(lm_dir_var,      dtype=float),
        "all_feature_direction_variance":   a(all_dir_var,     dtype=float),
        "landmark_feature_response_mean":   a(lm_resp_mean,    dtype=float),
        "landmark_feature_response_median": a(lm_resp_median,  dtype=float),
        "all_feature_response_mean":        a(all_resp_mean,   dtype=float),
        "all_feature_response_median":      a(all_resp_median, dtype=float),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_ALPHA   = 0.85
_LW      = 0.9
_GRID_KW = dict(linewidth=0.4, alpha=0.55, linestyle="--")


def _ax_setup(ax, ylabel: str, title: str) -> None:
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=3)
    ax.grid(True, which="major", **_GRID_KW)
    ax.grid(True, which="minor", linewidth=0.2, alpha=0.35, linestyle=":")
    ax.minorticks_on()
    ax.tick_params(which="major", labelsize=7, length=4)
    ax.tick_params(which="minor", labelsize=0, length=2)


def _plot_mean_median(
    ax,
    x,
    y_mean,
    y_median,
    ylabel: str,
    title: str,
    c_mean: str = "steelblue",
    c_median: str = "tomato",
) -> None:
    ax.plot(x, y_mean,   color=c_mean,   lw=_LW, alpha=_ALPHA, label="mean")
    ax.plot(x, y_median, color=c_median, lw=_LW, alpha=_ALPHA, label="median", linestyle="--")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.6)
    _ax_setup(ax, ylabel, title)


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(series: dict, title_prefix: str, figsize: tuple):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(7, 1, figsize=figsize, sharex=True)
    fig.suptitle(
        f"{title_prefix} — Per-frame Statistics",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )

    x = series["frame_ids"]

    # ── 1. Tracked landmarks ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(x, series["num_tracked_landmarks"],
            color="forestgreen", lw=_LW, alpha=_ALPHA)
    _ax_setup(ax, "count", "Tracked Landmarks")

    # ── 2. Reprojection error ─────────────────────────────────────────────
    _plot_mean_median(
        axes[1], x,
        series["landmark_reproj_error_px_mean"],
        series["landmark_reproj_error_px_median"],
        ylabel="[px]",
        title="Landmark Reprojection Error",
    )

    # ── 3. Parallax angle ─────────────────────────────────────────────────
    _plot_mean_median(
        axes[2], x,
        series["landmark_parallax_deg_mean"],
        series["landmark_parallax_deg_median"],
        ylabel="[deg]",
        title="Landmark Parallax Angle",
    )

    # ── 4. Landmark direction variance ────────────────────────────────────
    ax = axes[3]
    ax.plot(x, series["landmark_direction_variance"],
            color="mediumpurple", lw=_LW, alpha=_ALPHA)
    _ax_setup(ax, "1 − ‖μ‖", "Landmark Direction Variance")

    # ── 5. All-feature direction variance ────────────────────────────────
    ax = axes[4]
    ax.plot(x, series["all_feature_direction_variance"],
            color="darkorange", lw=_LW, alpha=_ALPHA)
    _ax_setup(ax, "1 − ‖μ‖", "All-Feature Direction Variance")

    # ── 6. Landmark feature response ─────────────────────────────────────
    _plot_mean_median(
        axes[5], x,
        series["landmark_feature_response_mean"],
        series["landmark_feature_response_median"],
        ylabel="response",
        title="Landmark Feature Response",
        c_mean="steelblue",
        c_median="tomato",
    )

    # ── 7. All-feature response ───────────────────────────────────────────
    _plot_mean_median(
        axes[6], x,
        series["all_feature_response_mean"],
        series["all_feature_response_median"],
        ylabel="response",
        title="All-Feature Response",
        c_mean="teal",
        c_median="salmon",
    )

    axes[-1].set_xlabel("Frame ID", fontsize=9)

    # ── X-axis ticks: auto-determine a round major interval ───────────────
    import numpy as np
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator
    if len(x) > 1:
        x_span = float(x[-1] - x[0])
        # Target ~8 major ticks across the span
        raw_step = x_span / 8.0
        # Round to a "nice" number (1, 2, 5, 10, 20, 50, 100, …)
        magnitude = 10 ** np.floor(np.log10(max(raw_step, 1.0)))
        nice_steps = [1, 2, 5, 10]
        major_step = min(
            (magnitude * s for s in nice_steps),
            key=lambda s: abs(s - raw_step),
        )
        major_step = max(1.0, float(major_step))
        minor_step = major_step / 5.0

        for ax in axes:
            ax.xaxis.set_major_locator(MultipleLocator(major_step))
            ax.xaxis.set_minor_locator(MultipleLocator(minor_step))
            # Only the bottom subplot shows tick labels (sharex)
        axes[-1].tick_params(axis="x", which="major", labelsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.997])
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Set matplotlib backend *before* importing pyplot
    import matplotlib
    if args.output_dir is not None:
        matplotlib.use("Agg")   # non-interactive; suitable for saving PNG
    # else: keep whatever default the environment provides (TkAgg / Qt / etc.)

    msgpack_path = Path(args.msgpack)
    if not msgpack_path.exists():
        print(f"[ERROR] File not found: {msgpack_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {msgpack_path} …")
    try:
        frames = load_frames(str(msgpack_path))
    except KeyError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  → {len(frames)} frames found in 'frames' section")

    series = build_series(frames)
    fig = build_figure(series, title_prefix=msgpack_path.stem, figsize=tuple(args.figsize))

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{msgpack_path.stem}_frame_statistics.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        import matplotlib.pyplot as plt
        plt.show()

    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    main()
