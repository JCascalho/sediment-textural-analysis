"""Publication-style plots for sediment textural analysis workbooks.

This script reads the Excel workbook produced by
``sediment_textural_moments.py`` and exports figures that
can replace manually prepared Excel plots.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd


PLOT_TYPES = (
    "probability-paper",
    "cumulative",
    "frequency",
    "histogram",
    "cumulative-linear",
    "excel-style",
    "scatter",
    "bivariate",
    "bivariate-fields",
    "all",
)
__version__ = "2.0.0"
DEFAULT_PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKBOOK_PATH = DEFAULT_PROJECT_DIR / "output_textural_parameters_Data_base_EDUCOAST.xlsx"
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_DIR / "figures"


@dataclass
class PlotConfig:
    """User-controlled plotting options."""

    workbook: Path = DEFAULT_WORKBOOK_PATH
    output_dir: Optional[Path] = DEFAULT_OUTPUT_DIR
    plot: str = "all"
    samples: str = "all"
    method: str = "graphical"
    x_parameter: str = "sorting"
    y_parameter: str = "mean_size_phi"
    color_parameter: Optional[str] = None
    figure_width: float = 11.69
    figure_height: float = 8.27
    dpi: int = 600
    format: str = "png"
    title: Optional[str] = None
    font_family: str = "Arial"
    title_size: float = 16.0
    label_size: float = 14.0
    tick_size: float = 12.0
    line_width: float = 1.4
    marker: str = "o"
    marker_size: float = 5.0
    colormap: str = "tab20"
    grid: str = "major_minor"
    legend: str = "outside"
    probability_min_percent: float = 0.02
    probability_max_percent: float = 99.98
    phi_min: Optional[float] = None
    phi_max: Optional[float] = None
    x_tick_step: float = 0.5
    frequency_y_max: Optional[float] = None
    reverse_phi_axis: bool = False
    show_mm_axis: bool = True


def read_config_file(path: Optional[Path]) -> dict:
    """Read a JSON style/configuration file."""
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_config(args: argparse.Namespace) -> PlotConfig:
    """Merge command-line options with an optional JSON configuration file."""
    file_values = read_config_file(args.config)
    cli_values = vars(args).copy()
    cli_values.pop("config", None)

    valid_names = {field.name for field in fields(PlotConfig)}
    merged = {key: value for key, value in file_values.items() if key in valid_names}
    for key, value in cli_values.items():
        if key in valid_names and value is not None:
            merged[key] = value

    if "workbook" not in merged:
        raise ValueError("--workbook is required, either on the command line or in the JSON config.")

    merged["workbook"] = Path(merged["workbook"])
    if merged.get("output_dir") is not None:
        merged["output_dir"] = Path(merged["output_dir"])
    return PlotConfig(**merged)


def phi_to_mm(phi):
    """Convert phi grain size to millimetres."""
    return 2 ** (-np.asarray(phi, dtype=float))


def probability_percent_ticks() -> tuple[list[float], list[float]]:
    """Return major and minor percentage ticks for probability-paper plots."""
    major = [
        0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 10, 16, 20, 25, 30, 40, 50,
        60, 70, 75, 80, 84, 90, 95, 96, 97, 98, 99, 99.5, 99.8, 99.9, 99.95, 99.98,
    ]
    minor = [
        0.02, 0.03, 0.04, 0.075, 0.15, 0.3, 0.4, 0.75, 1.5, 2.5, 3.5, 4.5,
        6, 7, 8, 9, 12, 14, 18, 22.5, 27.5, 32.5, 35, 37.5, 42.5, 45, 47.5,
        52.5, 55, 57.5, 62.5, 65, 67.5, 72.5, 77.5, 82, 86, 88, 91, 92, 93,
        94, 96.5, 97.5, 98.5, 99.25, 99.6, 99.7, 99.85, 99.90, 99.925, 99.96,
        99.97, 99.98,
    ]
    return major, minor


def probit_from_percent(percentages: list[float] | np.ndarray) -> np.ndarray:
    """Convert percentage values to normal-probability paper coordinates."""
    normal = NormalDist()
    probs = np.asarray(percentages, dtype=float) / 100.0
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    return np.array([normal.inv_cdf(float(p)) for p in probs], dtype=float)


def require_sheet(workbook: Path, sheet_name: str) -> pd.DataFrame:
    """Load a required sheet with a clear error message."""
    try:
        return pd.read_excel(workbook, sheet_name=sheet_name)
    except ValueError as exc:
        raise ValueError(
            f"Required sheet '{sheet_name}' was not found in {workbook}. "
            "Run sediment_textural_moments.py first."
        ) from exc


def selected_samples(df: pd.DataFrame, samples: str) -> list[str]:
    """Return selected sample names."""
    available = df["sample"].astype(str).drop_duplicates().tolist()
    if samples.strip().lower() == "all":
        return available
    wanted = [item.strip() for item in samples.split(",") if item.strip()]
    missing = [item for item in wanted if item not in available]
    if missing:
        raise ValueError(f"Sample(s) not found: {', '.join(missing)}")
    return wanted


def apply_common_style(plt, cfg: PlotConfig) -> None:
    """Apply global matplotlib style settings."""
    plt.rcParams.update({
        "font.family": cfg.font_family,
        "axes.titlesize": cfg.title_size,
        "axes.labelsize": cfg.label_size,
        "xtick.labelsize": cfg.tick_size,
        "ytick.labelsize": cfg.tick_size,
        "savefig.dpi": cfg.dpi,
    })


def set_phi_limits(ax, phi_values: np.ndarray, cfg: PlotConfig) -> None:
    """Apply phi-axis limits."""
    finite = np.asarray(phi_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if cfg.phi_min is not None and cfg.phi_max is not None:
        x_min, x_max = cfg.phi_min, cfg.phi_max
    elif finite.size:
        x_min = float(np.floor(finite.min() * 2.0) / 2.0) if cfg.phi_min is None else cfg.phi_min
        x_max = float(np.ceil(finite.max() * 2.0) / 2.0) if cfg.phi_max is None else cfg.phi_max
    else:
        x_min, x_max = -5.0, 5.0

    ax.set_xlim(x_max, x_min) if cfg.reverse_phi_axis else ax.set_xlim(x_min, x_max)


def apply_phi_ticks(ax, cfg: PlotConfig, minor_factor: int = 5) -> None:
    """Apply fixed phi tick spacing to the current x-axis limits."""
    step = float(cfg.x_tick_step)
    if not np.isfinite(step) or step <= 0:
        return

    x_min, x_max = sorted(ax.get_xlim())
    start = np.floor(x_min / step) * step
    stop = np.ceil(x_max / step) * step
    major_ticks = np.arange(start, stop + (step * 0.5), step)
    ax.set_xticks(major_ticks)

    if minor_factor and minor_factor > 1:
        minor_step = step / minor_factor
        minor_ticks = np.arange(start, stop + (minor_step * 0.5), minor_step)
        ax.set_xticks(minor_ticks, minor=True)


def add_grid(ax, cfg: PlotConfig) -> None:
    """Apply grid style."""
    if cfg.grid == "none":
        return
    ax.grid(True, which="major", linestyle="-", color="black", linewidth=0.8, alpha=0.75)
    if cfg.grid == "major_minor":
        ax.grid(True, which="minor", linestyle="--", color="gray", linewidth=0.45, alpha=0.65)


def setup_probability_axis(ax, phi_values: np.ndarray, cfg: PlotConfig) -> None:
    """Set probability-paper y axis and phi/mm x axes."""
    major_percent, minor_percent = probability_percent_ticks()
    ax.set_ylim(probit_from_percent([cfg.probability_min_percent, cfg.probability_max_percent]))
    ax.set_yticks(probit_from_percent(major_percent))
    ax.set_yticklabels(
        [f"{y:g}" for y in major_percent],
        fontweight="bold",
        fontsize=9,
    )
    ax.set_yticks(probit_from_percent(minor_percent), minor=True)

    set_phi_limits(ax, phi_values, cfg)
    x_min, x_max = sorted(ax.get_xlim())
    apply_phi_ticks(ax, cfg)
    ax.set_xlabel("Grain size (phi units)", fontweight="bold")
    ax.set_ylabel("Cumulative percentage (%)", fontweight="bold")
    add_grid(ax, cfg)

    if cfg.show_mm_axis:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        top_ticks = np.arange(np.ceil(x_min), np.floor(x_max) + 1.0, 1.0)
        ax_top.set_xticks(top_ticks)
        ax_top.set_xticklabels([f"{float(phi_to_mm(phi)):.3f}" for phi in top_ticks])
        ax_top.set_xlabel("Grain size (mm)", fontweight="bold")


def place_legend(ax, cfg: PlotConfig) -> None:
    """Place legend according to the selected option."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles or cfg.legend == "none":
        return
    if cfg.legend == "outside":
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=cfg.tick_size)
    elif cfg.legend == "bottom":
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=cfg.tick_size)
    else:
        ax.legend(loc="best", fontsize=cfg.tick_size)


def output_directory(cfg: PlotConfig) -> Path:
    """Return and create the output directory."""
    out = cfg.output_dir or cfg.workbook.with_suffix("").with_name(f"{cfg.workbook.stem}_plots")
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(fig, out_dir: Path, name: str, cfg: PlotConfig) -> Path:
    """Save one figure."""
    path = out_dir / f"{name}.{cfg.format.lower().lstrip('.')}"
    fig.tight_layout()
    save_format = cfg.format.lower().lstrip(".")
    try:
        fig.savefig(path, format=save_format, dpi=cfg.dpi, bbox_inches="tight")
        return path
    except OSError:
        for suffix in range(1, 100):
            fallback_path = out_dir / f"{name}_new_{suffix}.{save_format}"
            if not fallback_path.exists():
                fig.savefig(fallback_path, format=save_format, dpi=cfg.dpi, bbox_inches="tight")
                return fallback_path
        raise


def save_legend_figure(plt, handles, labels, out_dir: Path, name: str, cfg: PlotConfig) -> Path:
    """Save a standalone legend figure."""
    if not handles:
        return out_dir / f"{name}.{cfg.format.lower().lstrip('.')}"

    ncol = 4 if len(labels) > 20 else 2
    nrows = int(np.ceil(len(labels) / ncol))
    fig_width = max(6.0, ncol * 1.5)
    fig_height = max(3.0, nrows * 0.35)
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=True,
        fontsize=cfg.tick_size,
        title="Sample",
        title_fontsize=cfg.label_size,
    )
    return save_figure(fig, out_dir, name, cfg)


def plot_probability_paper(plt, plot_data: pd.DataFrame, out_dir: Path, cfg: PlotConfig) -> Path:
    """Export a blank probability-paper template."""
    phi_values = plot_data["phi"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
    setup_probability_axis(ax, phi_values, cfg)
    ax.set_title(cfg.title or "Probability Paper", fontweight="bold")
    return save_figure(fig, out_dir, "PROBABILITY_PAPER", cfg)


def plot_cumulative_curves(plt, plot_data: pd.DataFrame, out_dir: Path, cfg: PlotConfig) -> Path:
    """Export cumulative sample curves on probability paper."""
    samples = selected_samples(plot_data, cfg.samples)
    phi_values = plot_data["phi"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
    setup_probability_axis(ax, phi_values, cfg)
    colors = plt.colormaps[cfg.colormap]

    for i, sample in enumerate(samples):
        sample_df = plot_data[plot_data["sample"].astype(str) == sample].sort_values("phi")
        cumulative = np.minimum(sample_df["cumulative_percent"].to_numpy(dtype=float), cfg.probability_max_percent)
        valid = (cumulative >= cfg.probability_min_percent) & (cumulative <= cfg.probability_max_percent)
        if not np.any(valid):
            continue
        ax.plot(
            sample_df["phi"].to_numpy(dtype=float)[valid],
            probit_from_percent(cumulative[valid]),
            marker=cfg.marker,
            markersize=cfg.marker_size,
            linewidth=cfg.line_width,
            color=colors(i % colors.N),
            label=sample,
        )

    ax.set_title(cfg.title or "Cumulative Grain-size Curves", fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    save_legend_figure(plt, handles, labels, out_dir, "CUMULATIVE_CURVES_PROBABILITY_LEGEND", cfg)
    return save_figure(fig, out_dir, "CUMULATIVE_CURVES_PROBABILITY", cfg)


def plot_frequency_curves(plt, plot_data: pd.DataFrame, out_dir: Path, cfg: PlotConfig) -> list[Path]:
    """Export one frequency-distribution figure per selected sample."""
    paths = []
    samples = selected_samples(plot_data, cfg.samples)

    for sample in samples:
        sample_df = plot_data[plot_data["sample"].astype(str) == sample].sort_values("phi")
        fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
        ax.plot(
            sample_df["phi"],
            sample_df["frequency_percent"],
            marker=cfg.marker,
            markersize=cfg.marker_size,
            linewidth=cfg.line_width,
        )
        ax.fill_between(sample_df["phi"], sample_df["frequency_percent"], alpha=0.18)
        set_phi_limits(ax, sample_df["phi"].to_numpy(dtype=float), cfg)
        apply_phi_ticks(ax, cfg)
        if cfg.frequency_y_max is not None:
            ax.set_ylim(0, cfg.frequency_y_max)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel("Grain size (phi units)", fontweight="bold")
        ax.set_ylabel("Frequency (%)", fontweight="bold")
        ax.set_title(cfg.title or f"Frequency Distribution - {sample}", fontweight="bold")
        add_grid(ax, cfg)
        paths.append(save_figure(fig, out_dir, f"FREQUENCY_{safe_name(sample)}", cfg))
    return paths


def phi_step_width(phi_values: np.ndarray) -> float:
    """Estimate a useful bar width from the phi classes."""
    values = np.sort(np.asarray(phi_values, dtype=float))
    diffs = np.diff(values[np.isfinite(values)])
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.50
    return float(np.median(diffs))


def style_excel_axis(ax, cfg: PlotConfig) -> None:
    """Apply the heavier grid and axis style used by the Excel-like plots."""
    ax.grid(True, which="major", axis="y", linestyle="-", color="black", linewidth=0.8)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.tick_params(axis="both", colors="black", width=0.8)


def plot_histograms(plt, plot_data: pd.DataFrame, out_dir: Path, cfg: PlotConfig) -> list[Path]:
    """Export Excel-style relative-frequency histograms for selected samples."""
    paths = []
    samples = selected_samples(plot_data, cfg.samples)

    for sample in samples:
        sample_df = plot_data[plot_data["sample"].astype(str) == sample].sort_values("phi")
        phi = sample_df["phi"].to_numpy(dtype=float)
        frequency = sample_df["frequency_percent"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
        ax.bar(
            phi,
            frequency,
            width=phi_step_width(phi),
            color="#92d050",
            edgecolor="black",
            linewidth=0.8,
            align="center",
            zorder=2,
        )
        ax.plot(
            phi,
            frequency,
            color="black",
            marker="s",
            markersize=cfg.marker_size,
            linewidth=max(cfg.line_width, 1.8),
            zorder=3,
        )
        set_phi_limits(ax, phi, cfg)
        ax.set_xticks(phi)
        ax.set_xticklabels([f"{value:g}" for value in phi], fontweight="bold")
        ax.minorticks_off()
        if cfg.frequency_y_max is not None:
            ax.set_ylim(0, cfg.frequency_y_max)
        else:
            upper = max(10.0, float(np.nanmax(frequency)) if frequency.size else 10.0)
            ax.set_ylim(0, np.ceil((upper + 5.0) / 10.0) * 10.0)
        ax.set_xlabel("GRAIN SIZE (phi)", fontweight="bold")
        ax.set_ylabel("RELATIVE FREQUENCY (%)", fontweight="bold")
        ax.set_title(cfg.title or str(sample).upper(), fontweight="bold")
        style_excel_axis(ax, cfg)
        ax.tick_params(axis="x", labelrotation=90)
        paths.append(save_figure(fig, out_dir, f"HIST_FREQ_{safe_name(sample)}", cfg))
    return paths


def plot_cumulative_linear(plt, plot_data: pd.DataFrame, out_dir: Path, cfg: PlotConfig) -> list[Path]:
    """Export Excel-style linear cumulative-frequency plots for selected samples."""
    paths = []
    samples = selected_samples(plot_data, cfg.samples)

    for sample in samples:
        sample_df = plot_data[plot_data["sample"].astype(str) == sample].sort_values("phi")
        phi = sample_df["phi"].to_numpy(dtype=float)
        cumulative = sample_df["cumulative_percent"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
        ax.plot(
            phi,
            cumulative,
            color="black",
            marker="s",
            markersize=cfg.marker_size + 1,
            linewidth=max(cfg.line_width, 2.0),
            zorder=3,
        )
        set_phi_limits(ax, phi, cfg)
        apply_phi_ticks(ax, cfg)
        ax.set_ylim(0, 100)
        ax.set_xlabel("GRAIN SIZE (phi)", fontweight="bold")
        ax.set_ylabel("CUMULATIVE FREQUENCY (%)", fontweight="bold")
        ax.set_title(cfg.title or str(sample).upper(), fontweight="bold")
        style_excel_axis(ax, cfg)
        ax.tick_params(axis="x", labelrotation=45)
        paths.append(save_figure(fig, out_dir, f"CUMULATIVE_LINEAR_{safe_name(sample)}", cfg))
    return paths


def plot_excel_style(plt, plot_data: pd.DataFrame, out_dir: Path, cfg: PlotConfig) -> list[Path]:
    """Export both Excel-style histogram and linear cumulative plots."""
    saved = []
    saved.extend(plot_histograms(plt, plot_data, out_dir, cfg))
    plt.close("all")
    saved.extend(plot_cumulative_linear(plt, plot_data, out_dir, cfg))
    plt.close("all")
    return saved


def parameter_sheet_name(cfg: PlotConfig) -> str:
    """Return the parameter sheet selected for scatter plots."""
    if cfg.method == "moments":
        return "Moments_Parameters_phi"
    return "Graphical_Parameters_phi"


def pretty_parameter_label(parameter: str, method: str) -> str:
    """Return a readable axis label for a textural parameter."""
    labels = {
        "mean_size_phi": "Mean size (phi)",
        "mean_size_mm": "Mean size (mm)",
        "sorting": "Sorting (phi)",
        "standard_deviation": "Standard deviation (phi)",
        "skewness": "Skewness",
        "kurtosis": "Kurtosis",
    }
    if method == "moments" and parameter == "standard_deviation":
        return "Standard deviation (phi)"
    return labels.get(parameter, parameter.replace("_", " "))


def resolve_scatter_parameters(df: pd.DataFrame, cfg: PlotConfig) -> tuple[str, str]:
    """Map graphical defaults to the equivalent moments columns when needed."""
    x_parameter = cfg.x_parameter
    y_parameter = cfg.y_parameter
    if cfg.method == "moments" and x_parameter == "sorting" and "sorting" not in df.columns:
        x_parameter = "standard_deviation"
    return x_parameter, y_parameter


TEXTURAL_CLASS_LIMITS = {
    "graphical": {
        "mean_size_phi": [-4.00, -3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00],
        "sorting": [0.35, 0.50, 0.71, 1.00, 2.00, 4.00],
        "skewness": [-0.30, -0.10, 0.10, 0.30],
        "kurtosis": [0.67, 0.90, 1.11, 1.50, 3.00],
    },
    "moments": {
        "mean_size_phi": [-4.00, -3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00],
        "standard_deviation": [0.35, 0.50, 0.70, 1.00, 2.00, 4.00],
        "skewness": [-1.30, -0.43, 0.43, 1.30],
        "kurtosis": [1.70, 2.55, 3.70, 7.40],
    },
}


def add_textural_class_limits(ax, method: str, x_parameter: str, y_parameter: str) -> None:
    """Draw dashed class-limit lines for textural parameters."""
    limits = TEXTURAL_CLASS_LIMITS.get(method, {})

    x_min, x_max = sorted(ax.get_xlim())
    y_min, y_max = sorted(ax.get_ylim())

    for value in limits.get(x_parameter, []):
        if x_min <= value <= x_max:
            ax.axvline(
                value,
                color="red",
                linestyle="--",
                linewidth=4.0,
                alpha=0.9,
                zorder=1,
            )

    for value in limits.get(y_parameter, []):
        if y_min <= value <= y_max:
            ax.axhline(
                value,
                color="red",
                linestyle="--",
                linewidth=4.0,
                alpha=0.9,
                zorder=1,
            )


FIELD_CLASS_INTERVALS = {
    "graphical": {
        "mean_size_phi": [
            (-1.00, 0.00, "Vcs"),
            (0.00, 1.00, "Cs"),
            (1.00, 2.00, "Ms"),
            (2.00, 3.00, "Fs"),
            (3.00, 4.00, "Vfs"),
            (4.00, 5.00, "Vcsil"),
            (5.00, 6.00, "Csil"),
            (6.00, 7.00, "Msil"),
            (7.00, 8.00, "Fsil"),
            (8.00, 9.00, "Vfsil"),
            
        ],
        "sorting": [
            (-np.inf, 0.35, "VWSo"),
            (0.35, 0.50, "WSo"),
            (0.50, 0.71, "MWSo"),
            (0.71, 1.00, "MSo"),
            (1.00, 2.00, "PSo"),
            (2.00, 4.00, "VPSo"),
        ],
        "skewness": [
            (-np.inf, -0.30, "VCSk"),
            (-0.30, -0.10, "CSk"),
            (-0.10, 0.10, "SYM"),
            (0.10, 0.30, "FSk"),
            (0.30, np.inf, "VFSk"),
        ],
    },
    "moments": {
        "mean_size_phi": [
            (-1.00, 0.00, "Vcs"),
            (0.00, 1.00, "Cs"),
            (1.00, 2.00, "Ms"),
            (2.00, 3.00, "Fs"),
            (3.00, 4.00, "Vfs"),
            (4.00, 5.00, "Vcsil"),
            (5.00, 6.00, "Csil"),
            (6.00, 7.00, "Msil"),
            (7.00, 8.00, "Fsil"),
            (8.00, 9.00, "Vfsil"),
        ],
        "standard_deviation": [
            (-np.inf, 0.35, "VWSo"),
            (0.35, 0.50, "WSo"),
            (0.50, 0.70, "MWSo"),
            (0.70, 1.00, "MSo"),
            (1.00, 2.00, "PSo"),
            (2.00, 4.00, "VPSo"),
        ],
        "skewness": [
            (-np.inf, -1.30, "VCSk"),
            (-1.30, -0.43, "CSk"),
            (-0.43, 0.43, "SYM"),
            (0.43, 1.30, "FSk"),
            (1.30, np.inf, "VFSk"),
        ],
    },
}


def visible_class_segments(ax, method: str, parameter: str, axis: str) -> list[tuple[float, float, str]]:
    """Return class intervals clipped to the current visible axis range."""
    intervals = FIELD_CLASS_INTERVALS.get(method, {}).get(parameter, [])
    lower, upper = sorted(ax.get_xlim() if axis == "x" else ax.get_ylim())
    segments = []

    for start, stop, label in intervals:
        clipped_start = max(lower, start) if np.isfinite(start) else lower
        clipped_stop = min(upper, stop) if np.isfinite(stop) else upper
        if clipped_stop > clipped_start:
            segments.append((clipped_start, clipped_stop, label))

    return segments


def add_field_abbreviations(ax, method: str, x_parameter: str, y_parameter: str) -> None:
    """Place field abbreviations inside the visible class domains."""
    x_segments = visible_class_segments(ax, method, x_parameter, "x")
    y_segments = visible_class_segments(ax, method, y_parameter, "y")

    if x_segments and y_segments:
        for x_start, x_stop, x_label in x_segments:
            for y_start, y_stop, y_label in y_segments:
                width = x_stop - x_start
                height = y_stop - y_start
                if width <= 0 or height <= 0:
                    continue
                ax.text(
                    (x_start + x_stop) / 2.0,
                    (y_start + y_stop) / 2.0,
                    f"{y_label}\n{x_label}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="black",
                    alpha=0.65,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=1.2),
                    zorder=2,
                )
        return

    for x_start, x_stop, label in x_segments:
        ax.text(
            (x_start + x_stop) / 2.0,
            0.97,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="black",
            alpha=0.75,
            zorder=2,
        )

    for y_start, y_stop, label in y_segments:
        ax.text(
            0.02,
            (y_start + y_stop) / 2.0,
            label,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="black",
            alpha=0.75,
            zorder=2,
        )


def plot_scatter_dataframe(
    plt,
    df: pd.DataFrame,
    out_dir: Path,
    cfg: PlotConfig,
    method: str,
    x_parameter: str,
    y_parameter: str,
    color_parameter: Optional[str] = None,
    title: Optional[str] = None,
    output_name: Optional[str] = None,
) -> Path:
    """Export one scatter plot from an already loaded parameter table."""
    samples = selected_samples(df, cfg.samples)
    df = df[df["sample"].astype(str).isin(samples)].copy()

    missing = [col for col in [x_parameter, y_parameter] if col not in df.columns]
    if missing:
        raise ValueError(f"Parameter column(s) not found for scatter plot: {', '.join(missing)}")

    fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
    color_values = None
    if color_parameter:
        if color_parameter not in df.columns:
            raise ValueError(f"Color parameter not found: {color_parameter}")
        color_values = pd.to_numeric(df[color_parameter], errors="coerce")

    is_mean = df["sample"].astype(str).str.upper().eq("MEAN")
    regular_df = df[~is_mean]
    mean_df = df[is_mean]

    scatter = ax.scatter(
        pd.to_numeric(regular_df[x_parameter], errors="coerce"),
        pd.to_numeric(regular_df[y_parameter], errors="coerce"),
        c=color_values[~is_mean] if color_values is not None else None,
        cmap=cfg.colormap if color_values is not None else None,
        s=cfg.marker_size * 18,
        marker=cfg.marker,
        facecolor="tab:blue" if color_values is None else None,
        edgecolor="black",
        linewidth=0.5,
    )

    if not mean_df.empty:
        ax.scatter(
            pd.to_numeric(mean_df[x_parameter], errors="coerce"),
            pd.to_numeric(mean_df[y_parameter], errors="coerce"),
            s=cfg.marker_size * 35,
            marker="*",
            facecolor="red",
            edgecolor="black",
            linewidth=0.7,
            zorder=8,
        )
    label_offsets = [
        (4, 4),
        (-4, 4),
        (4, -4),
        (-4, -4),
        (6, 0),
        (-6, 0),
        (0, 6),
        (0, -6),
        (6, 3),
        (-6, 3),
        (6, -3),
        (-6, -3),
        ]

  
    mean_label_offset = (-8, -2)

    for i, (_, row) in enumerate(df.iterrows()):
        sample_label = str(row["sample"])

        if sample_label.upper() == "MEAN":
            dx, dy = mean_label_offset
        else:
            dx, dy = label_offsets[i % len(label_offsets)]

        ax.annotate(
           sample_label,
           (row[x_parameter], row[y_parameter]),
           xytext=(dx, dy),
           textcoords="offset points",
           fontsize=13 if sample_label.upper() == "MEAN" else 12,
           fontweight="bold",
           color="red" if sample_label.upper() == "MEAN" else "black",
           ha="left" if dx >= 0 else "right",
           va="bottom" if dy >= 0 else "top",
           bbox=dict(
               facecolor="white",
               edgecolor="none",
               alpha=0.65,
               pad=0.4,
           ),
           zorder=9 if sample_label.upper() == "MEAN" else 6,
      )

     
  
    if color_values is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(pretty_parameter_label(color_parameter, method))
    ax.set_xlabel(pretty_parameter_label(x_parameter, method), fontweight="bold")
    ax.set_ylabel(pretty_parameter_label(y_parameter, method), fontweight="bold")
    ax.set_title(title or f"{pretty_parameter_label(y_parameter, method)} vs {pretty_parameter_label(x_parameter, method)}", fontweight="bold")
    add_textural_class_limits(ax, method, x_parameter, y_parameter)
    add_grid(ax, cfg)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    name = output_name or f"SCATTER_{method}_{y_parameter}_vs_{x_parameter}"
    return save_figure(fig, out_dir, name, cfg)


def plot_bivariate_field_dataframe(
    plt,
    df: pd.DataFrame,
    out_dir: Path,
    cfg: PlotConfig,
    method: str,
    x_parameter: str,
    y_parameter: str,
    title: Optional[str],
    output_name: str,
) -> Path:
    """Export one bivariate plot with sample points and labelled class domains."""
    samples = selected_samples(df, cfg.samples)
    df = df[df["sample"].astype(str).isin(samples)].copy()
    df = df[~df["sample"].astype(str).str.upper().eq("MEAN")]

    missing = [col for col in [x_parameter, y_parameter] if col not in df.columns]
    if missing:
        raise ValueError(f"Parameter column(s) not found for bivariate field plot: {', '.join(missing)}")

    fig, ax = plt.subplots(figsize=(cfg.figure_width, cfg.figure_height))
    ax.scatter(
        pd.to_numeric(df[x_parameter], errors="coerce"),
        pd.to_numeric(df[y_parameter], errors="coerce"),
        s=cfg.marker_size * 18,
        marker=cfg.marker,
        facecolor="tab:blue",
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
    )

    ax.set_xlabel(pretty_parameter_label(x_parameter, method), fontweight="bold")
    ax.set_ylabel(pretty_parameter_label(y_parameter, method), fontweight="bold")
    ax.set_title(title or f"{pretty_parameter_label(y_parameter, method)} vs {pretty_parameter_label(x_parameter, method)}", fontweight="bold")
    add_textural_class_limits(ax, method, x_parameter, y_parameter)
    add_field_abbreviations(ax, method, x_parameter, y_parameter)
    add_grid(ax, cfg)

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    return save_figure(fig, out_dir, output_name, cfg)


def plot_parameter_scatter(plt, workbook: Path, out_dir: Path, cfg: PlotConfig) -> Path:
    """Export a user-selected scatter plot from the graphical or moments parameter sheet."""
    df = require_sheet(workbook, parameter_sheet_name(cfg))
    x_parameter, y_parameter = resolve_scatter_parameters(df, cfg)
    return plot_scatter_dataframe(
        plt,
        df,
        out_dir,
        cfg,
        cfg.method,
        x_parameter,
        y_parameter,
        color_parameter=cfg.color_parameter,
        title=cfg.title,
    )


def bivariate_specs(method: str) -> list[tuple[str, str, Optional[str], str]]:
    """Return the selected bivariate plots for one calculation method."""
    if method == "moments":
        spread = "standard_deviation"
        prefix = "MOMENTS"
    else:
        spread = "sorting"
        prefix = "GRAPHICAL"

    return [
        (spread, "mean_size_phi", None, f"{prefix}_MEAN_SIZE_vs_SPREAD"),
        (spread, "skewness", None, f"{prefix}_SKEWNESS_vs_SPREAD"),
    ]


def plot_bivariate_suite(plt, workbook: Path, out_dir: Path, cfg: PlotConfig) -> list[Path]:
    """Export standard bivariate plots for graphical and moments results."""
    saved: list[Path] = []
    method_sheets = [
        ("graphical", "Graphical_Parameters_phi"),
        ("moments", "Moments_Parameters_phi"),
    ]

    for method, sheet_name in method_sheets:
        try:
            df = require_sheet(workbook, sheet_name)
        except ValueError:
            continue

        for x_parameter, y_parameter, color_parameter, output_name in bivariate_specs(method):
            title = (
                f"{pretty_parameter_label(y_parameter, method)} vs "
                f"{pretty_parameter_label(x_parameter, method)} - {method.capitalize()} method"
            )
            saved.append(plot_scatter_dataframe(
                plt,
                df,
                out_dir,
                cfg,
                method,
                x_parameter,
                y_parameter,
                color_parameter=color_parameter,
                title=title,
                output_name=output_name,
            ))
            plt.close("all")

    return saved


def plot_bivariate_field_suite(plt, workbook: Path, out_dir: Path, cfg: PlotConfig) -> list[Path]:
    """Export bivariate plots with only points, class limits, and field abbreviations."""
    saved: list[Path] = []
    method_sheets = [
        ("graphical", "Graphical_Parameters_phi"),
        ("moments", "Moments_Parameters_phi"),
    ]

    for method, sheet_name in method_sheets:
        try:
            df = require_sheet(workbook, sheet_name)
        except ValueError:
            continue

        for x_parameter, y_parameter, _color_parameter, output_name in bivariate_specs(method):
            title = (
                f"{pretty_parameter_label(y_parameter, method)} vs "
                f"{pretty_parameter_label(x_parameter, method)} - {method.capitalize()} method"
            )
            saved.append(plot_bivariate_field_dataframe(
                plt,
                df,
                out_dir,
                cfg,
                method,
                x_parameter,
                y_parameter,
                title=title,
                output_name=f"{output_name}_FIELDS",
            ))
            plt.close("all")

    return saved


def safe_name(value: str) -> str:
    """Make a safe filename fragment."""
    keep = [char if char.isalnum() or char in ("-", "_") else "_" for char in str(value)]
    return "".join(keep).strip("_") or "sample"


def run(cfg: PlotConfig) -> list[Path]:
    """Create the requested plots."""
    try:
        import matplotlib
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib is required for figure export. Install the plotting dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_common_style(plt, cfg)
    plot_data = require_sheet(cfg.workbook, "Plot_Data_Long")
    out_dir = output_directory(cfg)
    saved: list[Path] = []

    plot_request = cfg.plot.lower()
    if plot_request in {"probability-paper", "all"}:
        saved.append(plot_probability_paper(plt, plot_data, out_dir, cfg))
        plt.close("all")
    if plot_request in {"cumulative", "all"}:
        saved.append(plot_cumulative_curves(plt, plot_data, out_dir, cfg))
        plt.close("all")
    if plot_request in {"frequency"}:
        saved.extend(plot_frequency_curves(plt, plot_data, out_dir, cfg))
        plt.close("all")
    if plot_request in {"histogram"}:
        saved.extend(plot_histograms(plt, plot_data, out_dir, cfg))
        plt.close("all")
    if plot_request in {"cumulative-linear"}:
        saved.extend(plot_cumulative_linear(plt, plot_data, out_dir, cfg))
        plt.close("all")
    if plot_request in {"excel-style", "all"}:
        saved.extend(plot_excel_style(plt, plot_data, out_dir, cfg))
        plt.close("all")
    if plot_request in {"scatter"}:
        saved.append(plot_parameter_scatter(plt, cfg.workbook, out_dir, cfg))
        plt.close("all")
    if plot_request in {"bivariate", "all"}:
        saved.extend(plot_bivariate_suite(plt, cfg.workbook, out_dir, cfg))
        plt.close("all")
    if plot_request in {"bivariate-fields", "all"}:
        saved.extend(plot_bivariate_field_suite(plt, cfg.workbook, out_dir, cfg))
        plt.close("all")

    return saved


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description="Create sediment textural analysis plots from an output workbook.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON file containing plot options.")
    parser.add_argument("--workbook", "-i", type=Path, default=DEFAULT_WORKBOOK_PATH, help="Output workbook from the analysis routine.")
    parser.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where figures will be saved.")
    parser.add_argument("--plot", choices=PLOT_TYPES, default=None, help="Plot type to export. Use 'excel-style' for histogram plus linear cumulative plots.")
    parser.add_argument("--samples", default=None, help="Comma-separated sample names, or 'all'.")
    parser.add_argument("--method", choices=["graphical", "moments"], default=None, help="Parameter sheet used for scatter plots.")
    parser.add_argument("--x-parameter", default=None, help="Custom scatter x-axis parameter column. Default: sorting for graphical, standard_deviation for moments.")
    parser.add_argument("--y-parameter", default=None, help="Custom scatter y-axis parameter column. Default: mean_size_phi.")
    parser.add_argument("--color-parameter", default=None, help="Optional numeric parameter used to color scatter points.")
    parser.add_argument("--figure-width", type=float, default=None, help="Figure width in inches.")
    parser.add_argument("--figure-height", type=float, default=None, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=None, help="Raster export resolution.")
    parser.add_argument("--format", choices=["jpg", "jpeg", "png", "pdf", "svg", "tif", "tiff"], default=None, help="Figure format.")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument("--font-family", default=None, help="Font family.")
    parser.add_argument("--title-size", type=float, default=None, help="Title font size.")
    parser.add_argument("--label-size", type=float, default=None, help="Axis-label font size.")
    parser.add_argument("--tick-size", type=float, default=None, help="Tick-label font size.")
    parser.add_argument("--line-width", type=float, default=None, help="Line width.")
    parser.add_argument("--marker", default=None, help="Matplotlib marker symbol.")
    parser.add_argument("--marker-size", type=float, default=None, help="Marker size.")
    parser.add_argument("--colormap", default=None, help="Matplotlib colormap name.")
    parser.add_argument("--grid", choices=["none", "major", "major_minor"], default=None, help="Grid style.")
    parser.add_argument("--legend", choices=["none", "inside", "outside", "bottom"], default=None, help="Legend placement.")
    parser.add_argument("--probability-min-percent", type=float, default=None, help="Minimum probability-paper percent.")
    parser.add_argument("--probability-max-percent", type=float, default=None, help="Maximum probability-paper percent.")
    parser.add_argument("--phi-min", type=float, default=None, help="Minimum phi axis value.")
    parser.add_argument("--phi-max", type=float, default=None, help="Maximum phi axis value.")
    parser.add_argument("--x-tick-step", type=float, default=None, help="Major x-axis tick spacing for phi plots. Default: 0.5.")
    parser.add_argument("--frequency-y-max", type=float, default=None, help="Frequency plot y-axis maximum.")
    parser.add_argument("--reverse-phi-axis", action="store_true", default=None, help="Show coarser sizes on the left.")
    parser.add_argument("--no-mm-axis", dest="show_mm_axis", action="store_false", default=None, help="Hide top millimetre axis.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point."""
    cfg = merge_config(build_parser().parse_args(argv))
    saved = run(cfg)
    for path in saved:
        print(f"Plot saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
