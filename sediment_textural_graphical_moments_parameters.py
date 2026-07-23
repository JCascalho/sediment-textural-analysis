"""Sediment textural analysis using graphical and moments methods.

This command-line tool refactors the older Folk and Ward graphical routine and
the older moments routine into one reproducible workflow.

The input table follows the architecture used by ``sediment_textural_analysis.py``:

- one row per sample;
- one sample identifier column;
- numeric grain-size class columns as phi or millimetres;
- optional x/y coordinate columns for STA-ready export.

Frequencies are normalized internally, so sample rows may sum to 100, 1, or any
positive value. Grain-size class headers can be supplied in phi or millimetres;
with ``--units auto`` the script detects millimetres and converts them to phi.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd

__version__ = "2.0.0"
INPUT_PREFIX = "INPUT_"
OUTPUT_PREFIX = "OUTPUT_"
PERCENTILES = (5, 16, 25, 50, 75, 84, 95)
DEFAULT_PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_PATH = DEFAULT_PROJECT_DIR / "Data_base_EDUCOAST.xlsx"
DEFAULT_OUTPUT_PATH = DEFAULT_PROJECT_DIR / "output_textural_parameters_Data_base_EDUCOAST.xlsx"


@dataclass
class Config:
    """Runtime settings for sediment textural analysis."""

    method: str = "graphical"
    units: str = "auto"
    sheet_name: Optional[str] = None
    sample_column: Optional[str] = None
    cumulative_max_percent: float = 99.98
    monotonicity_tolerance: float = 1e-8
    plot_probability_paper: bool = False
    plot_cumulative_curves: bool = False
    plot_output_dir: Optional[Path] = None
    plot_format: str = "jpg"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip accidental leading/trailing spaces from column names."""
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def read_input_table(input_path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Read CSV or Excel input."""
    suffix = input_path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(input_path)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(input_path, sheet_name=sheet_name or 0)
    raise ValueError(f"Unsupported input file type: {input_path.suffix}")


def default_output_path(input_path: Path, user_output: Optional[str] = None) -> Path:
    """Create a logical output path when the user does not provide one."""
    if user_output:
        return Path(user_output)

    if input_path.name.startswith(INPUT_PREFIX):
        name = input_path.name.replace(INPUT_PREFIX, OUTPUT_PREFIX, 1)
        name = Path(name).with_suffix(".xlsx").name
        if "TEXTURAL" not in name.upper():
            name = name.replace(".xlsx", "_TEXTURAL_PARAMETERS.xlsx")
    else:
        name = f"{OUTPUT_PREFIX}{input_path.stem}_TEXTURAL_PARAMETERS.xlsx"
    return input_path.with_name(name)


def optional_path(value: Optional[str]) -> Optional[Path]:
    """Parse optional path arguments."""
    if value is None or str(value).strip() == "":
        return None
    return Path(value)


def mm_to_phi(mm: np.ndarray) -> np.ndarray:
    """Convert grain size from millimetres to phi units."""
    mm = np.asarray(mm, dtype=float)
    if np.any(mm <= 0):
        raise ValueError("Millimetre grain-size classes must be positive.")
    return -np.log2(mm)


def phi_to_mm(phi):
    """Convert phi grain size to millimetres."""
    return 2 ** (-phi)


def looks_like_mm(values: np.ndarray) -> bool:
    """Heuristic to infer whether numeric grain-size headers are millimetres."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return False
    return bool(np.all(finite > 0) and finite.max() <= 64 and finite.min() >= 1e-4)


def find_any_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find a likely matching column by exact or partial case-insensitive match."""
    lowered = {col: str(col).strip().lower() for col in df.columns}
    wanted = [name.strip().lower() for name in candidates]

    for col, low in lowered.items():
        if low in wanted:
            return col

    for col, low in lowered.items():
        for candidate in wanted:
            if candidate and candidate in low:
                return col
    return None


def parse_grain_size_columns(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Return columns whose names are numeric grain-size classes."""
    size_values = pd.to_numeric(pd.Index(df.columns), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(size_values)
    columns = [col for col, ok in zip(df.columns, mask) if ok]
    values = size_values[mask]
    if len(columns) == 0:
        raise ValueError(
            "No numeric grain-size class columns were found. "
            "Use numeric headers such as -1, 0, 1, 2 for phi or 2, 1, 0.5, 0.25 for mm."
        )
    return columns, values


def choose_sample_column(df: pd.DataFrame, grain_size_columns: list[str], sample_column: Optional[str]) -> str:
    """Choose the sample identifier column."""
    if sample_column:
        if sample_column not in df.columns:
            raise ValueError(f"Sample column not found: {sample_column}")
        return sample_column

    non_size = [col for col in df.columns if col not in set(grain_size_columns)]
    if not non_size:
        raise ValueError("No sample identifier column found.")
    return non_size[0]


def normalize_frequencies(frequencies: pd.Series) -> tuple[pd.Series, float]:
    """Return non-negative percentage frequencies and the raw total."""
    freq = pd.to_numeric(frequencies, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(freq.sum())
    if total <= 0:
        return freq * np.nan, total
    return (freq / total) * 100.0, total


class FolkWardAnalyzer:
    """Folk and Ward (1957) graphical textural statistics."""

    sorting_thresholds = [
        (0.35, "Very well sorted"),
        (0.50, "Well sorted"),
        (0.71, "Moderately well sorted"),
        (1.00, "Moderately sorted"),
        (2.00, "Poorly sorted"),
        (4.00, "Very poorly sorted"),
        (float("inf"), "Extremely poorly sorted"),
    ]

    skewness_thresholds = [
        (-0.30, "Very coarse skewed"),
        (-0.10, "Coarse skewed"),
        (0.10, "Symmetrical"),
        (0.30, "Fine skewed"),
        (float("inf"), "Very fine skewed"),
    ]

    kurtosis_thresholds = [
        (0.67, "Very platykurtic"),
        (0.90, "Platykurtic"),
        (1.11, "Mesokurtic"),
        (1.50, "Leptokurtic"),
        (3.00, "Very leptokurtic"),
        (float("inf"), "Extremely leptokurtic"),
    ]

    def __init__(self, cumulative_max_percent: float = 99.98) -> None:
        self.cumulative_max_percent = float(cumulative_max_percent)

    @staticmethod
    def mean_size(phi16: float, phi50: float, phi84: float) -> float:
        """Folk and Ward mean grain size."""
        return (phi16 + phi50 + phi84) / 3.0

    @staticmethod
    def sorting(phi5: float, phi16: float, phi84: float, phi95: float) -> float:
        """Inclusive graphic standard deviation."""
        return ((phi84 - phi16) / 4.0) + ((phi95 - phi5) / 6.6)

    @staticmethod
    def skewness(phi5: float, phi16: float, phi50: float, phi84: float, phi95: float) -> float:
        """Inclusive graphic skewness."""
        term1 = (
            (phi84 + phi16 - 2.0 * phi50) / (2.0 * (phi84 - phi16))
            if (phi84 - phi16) != 0
            else np.nan
        )
        term2 = (
            (phi95 + phi5 - 2.0 * phi50) / (2.0 * (phi95 - phi5))
            if (phi95 - phi5) != 0
            else np.nan
        )
        return float(term1 + term2)

    @staticmethod
    def kurtosis(phi5: float, phi25: float, phi75: float, phi95: float) -> float:
        """Graphic kurtosis."""
        iqr = phi75 - phi25
        denom = 2.44 * iqr if iqr != 0 else np.nan
        return float((phi95 - phi5) / denom)

    @staticmethod
    def classify_grain_size(mean_phi: float) -> str:
        """Classify mean grain size in phi units."""
        if not np.isfinite(mean_phi):
            return "N/A"
        if mean_phi > 8:
            return "Clay"
        if mean_phi > 7:
            return "Fine silt"
        if mean_phi > 6:
            return "Medium silt"
        if mean_phi > 5:
            return "Coarse silt"
        if mean_phi > 4:
            return "Silt"
        if mean_phi > 3:
            return "Very fine sand"
        if mean_phi > 2:
            return "Fine sand"
        if mean_phi > 1:
            return "Medium sand"
        if mean_phi > 0:
            return "Coarse sand"
        if mean_phi > -1:
            return "Very coarse sand"
        if mean_phi > -2:
            return "Fine pebbles/granules"
        if mean_phi > -3:
            return "Medium pebbles"
        if mean_phi > -4:
            return "Coarse pebbles"
        if mean_phi > -5:
            return "very coarse pebbles"
        return "Cobbles"

    @staticmethod
    def classify_from_thresholds(value: float, thresholds: list[tuple[float, str]]) -> str:
        """Classify a numeric value from upper-bound thresholds."""
        if not np.isfinite(value):
            return "N/A"
        for upper, label in thresholds:
            if value <= upper:
                return label
        return "N/A"

    def calculate_percentiles(self, phi_values: np.ndarray, cumulative_percent: np.ndarray) -> dict[str, float]:
        """Interpolate Folk and Ward percentiles from the cumulative curve."""
        clipped = np.minimum(cumulative_percent, self.cumulative_max_percent)
        return {f"phi{p}": float(np.interp(p, clipped, phi_values)) for p in PERCENTILES}

    def analyze_sample(self, phi_values: np.ndarray, frequencies: pd.Series) -> dict:
        """Analyze one sample frequency distribution."""
        freq_percent, total = normalize_frequencies(frequencies)

        if total <= 0:
            percentiles = {f"phi{p}": np.nan for p in PERCENTILES}
            return {
                **percentiles,
                "mean_size_phi": np.nan,
                "mean_size_mm": np.nan,
                "sorting": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "grain_size_class": "N/A",
                "sorting_class": "N/A",
                "skewness_class": "N/A",
                "kurtosis_class": "N/A",
                "total_raw": total,
                "_cum_raw": None,
                "_cum_clean": None,
            }

        cumulative_raw = freq_percent.cumsum()
        cumulative_clean = np.maximum.accumulate(cumulative_raw.to_numpy(dtype=float))
        cumulative_clean = np.clip(cumulative_clean, 0.0, 100.0)
        percentiles = self.calculate_percentiles(phi_values, cumulative_clean)

        phi5 = percentiles["phi5"]
        phi16 = percentiles["phi16"]
        phi25 = percentiles["phi25"]
        phi50 = percentiles["phi50"]
        phi75 = percentiles["phi75"]
        phi84 = percentiles["phi84"]
        phi95 = percentiles["phi95"]
        values = np.array([phi5, phi16, phi25, phi50, phi75, phi84, phi95], dtype=float)

        if np.isnan(values).any():
            mean_phi = sorting = skewness = kurtosis = np.nan
        else:
            mean_phi = self.mean_size(phi16, phi50, phi84)
            sorting = self.sorting(phi5, phi16, phi84, phi95)
            skewness = self.skewness(phi5, phi16, phi50, phi84, phi95)
            kurtosis = self.kurtosis(phi5, phi25, phi75, phi95)

        return {
            **percentiles,
            "mean_size_phi": mean_phi,
            "mean_size_mm": phi_to_mm(mean_phi) if np.isfinite(mean_phi) else np.nan,
            "sorting": sorting,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "grain_size_class": self.classify_grain_size(mean_phi),
            "sorting_class": self.classify_from_thresholds(sorting, self.sorting_thresholds),
            "skewness_class": self.classify_from_thresholds(skewness, self.skewness_thresholds),
            "kurtosis_class": self.classify_from_thresholds(kurtosis, self.kurtosis_thresholds),
            "total_raw": total,
            "_cum_raw": cumulative_raw.to_numpy(dtype=float),
            "_cum_clean": cumulative_clean,
        }


class MomentsAnalyzer:
    """Moment statistics adapted from the older ``Python_moments.py`` routine."""

    standard_deviation_thresholds = [
        (0.35, "Very well sorted"),
        (0.50, "Well sorted"),
        (0.70, "Moderately well sorted"),
        (1.00, "Moderately sorted"),
        (2.00, "Poorly sorted"),
        (4.00, "Very poorly sorted"),
        (float("inf"), "Extremely poorly sorted"),
    ]

    skewness_thresholds = [
        (-1.30, "Very coarse skewed"),
        (-0.43, "Coarse skewed"),
        (0.43, "Symmetrical"),
        (1.30, "Fine skewed"),
        (float("inf"), "Very fine skewed"),
    ]

    kurtosis_thresholds = [
        (1.70, "Very platykurtic"),
        (2.55, "Platykurtic"),
        (3.70, "Mesokurtic"),
        (7.40, "Leptokurtic"),
        (float("inf"), "Very leptokurtic"),
    ]

    @staticmethod
    def classify_from_thresholds(value: float, thresholds: list[tuple[float, str]]) -> str:
        """Classify a numeric value from upper-bound thresholds."""
        if not np.isfinite(value):
            return "N/A"
        for upper, label in thresholds:
            if value < upper:
                return label
        return "N/A"

    @staticmethod
    def classify_skewness(value: float) -> str:
        """Classify moment skewness using the original script's boundaries."""
        if not np.isfinite(value):
            return "N/A"
        if value > 1.30:
            return "Very fine skewed"
        if value > 0.43:
            return "Fine skewed"
        if value > -0.43:
            return "Symmetrical"
        if value > -1.30:
            return "Coarse skewed"
        return "Very coarse skewed"

    def analyze_sample(self, phi_values: np.ndarray, frequencies: pd.Series) -> dict:
        """Analyze one sample using normalized moment formulas.

        This is the same formula family used in the old script:
        mean = sum(percent * phi) / 100;
        variance = sum(percent * (phi - mean)^2) / 100;
        skewness and kurtosis use the third and fourth standardized moments.
        """
        freq_percent, total = normalize_frequencies(frequencies)

        if total <= 0:
            return {
                "mean_size_phi": np.nan,
                "mean_size_mm": np.nan,
                "standard_deviation": np.nan,
                "variance": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "grain_size_class": "N/A",
                "standard_deviation_class": "N/A",
                "skewness_class": "N/A",
                "kurtosis_class": "N/A",
                "total_raw": total,
            }

        weights = freq_percent.to_numpy(dtype=float)
        phi = np.asarray(phi_values, dtype=float)
        mean_phi = float((weights * phi).sum() / 100.0)
        variance = float((weights * ((phi - mean_phi) ** 2)).sum() / 100.0)
        standard_deviation = float(np.sqrt(variance)) if variance >= 0 else np.nan

        if np.isfinite(standard_deviation) and standard_deviation != 0:
            skewness = float((weights * ((phi - mean_phi) ** 3)).sum() / (100.0 * standard_deviation**3))
            kurtosis = float((weights * ((phi - mean_phi) ** 4)).sum() / (100.0 * standard_deviation**4))
        else:
            skewness = np.nan
            kurtosis = np.nan

        return {
            "mean_size_phi": mean_phi,
            "mean_size_mm": phi_to_mm(mean_phi) if np.isfinite(mean_phi) else np.nan,
            "standard_deviation": standard_deviation,
            "variance": variance,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "grain_size_class": FolkWardAnalyzer.classify_grain_size(mean_phi),
            "standard_deviation_class": self.classify_from_thresholds(
                standard_deviation, self.standard_deviation_thresholds
            ),
            "skewness_class": self.classify_skewness(skewness),
            "kurtosis_class": self.classify_from_thresholds(kurtosis, self.kurtosis_thresholds),
            "total_raw": total,
        }


def create_reference_tables() -> dict[str, pd.DataFrame]:
    """Create reference sheets for the output workbook."""
    phi_scale = pd.DataFrame({
        "Phi": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        "Grain_size_mm": [32, 16, 8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625, 0.031, 0.016, 0.008, 0.004],
        "Sediment_class": [
            "Cobbles",
            "Very coarse pebbles",
            "Coarse pebbles",
            "Medium pebbles",
            "Fine pebbles/granules",
            "Very coarse sand",
            "Coarse sand",
            "Medium sand",
            "Fine sand",
            "Very fine sand",
            "Very coarse silt",
            "Coarse silt",
            "Medium silt",
            "Fine silt",
        ],
    })

    graphical_criteria = pd.DataFrame({
        "Parameter": ["Mean size (phi)", "Sorting (sigmaI)", "Skewness (SkI)", "Kurtosis (KG)"],
        "Classification_ranges": [
            "Clay (>8), Silt (4 to 8), very fine sand (3 to 4), fine sand (2 to 3), medium sand (1 to 2), coarse sand (0 to 1), very coarse sand (-1 to 0), gravel (<-1)",
            "Very well sorted (<0.35), well sorted (0.35 to 0.50), moderately well sorted (0.50 to 0.71), moderately sorted (0.71 to 1.00), poorly sorted (1.00 to 2.00), very poorly sorted (2.00 to 4.00), extremely poorly sorted (>4.00)",
            "Very coarse skewed (<-0.30), coarse skewed (-0.30 to -0.10), symmetrical (-0.10 to 0.10), fine skewed (0.10 to 0.30), very fine skewed (>0.30)",
            "Very platykurtic (<0.67), platykurtic (0.67 to 0.90), mesokurtic (0.90 to 1.11), leptokurtic (1.11 to 1.50), very leptokurtic (1.50 to 3.00), extremely leptokurtic (>3.00)",
        ],
        "Formula": [
            "(phi16 + phi50 + phi84) / 3",
            "(phi84 - phi16) / 4 + (phi95 - phi5) / 6.6",
            "(phi84 + phi16 - 2*phi50) / (2*(phi84 - phi16)) + (phi95 + phi5 - 2*phi50) / (2*(phi95 - phi5))",
            "(phi95 - phi5) / (2.44 * (phi75 - phi25))",
        ],
    })

    moments_criteria = pd.DataFrame({
        "Parameter": ["Mean size (phi)", "Standard deviation", "Skewness", "Kurtosis"],
        "Classification_ranges": [
            "Same mean grain-size classes as graphical method.",
            "Very well sorted (<0.35), well sorted (0.35 to 0.50), moderately well sorted (0.50 to 0.70), moderately sorted (0.70 to 1.00), poorly sorted (1.00 to 2.00), very poorly sorted (2.00 to 4.00), extremely poorly sorted (>4.00)",
            "Very coarse skewed (<-1.30), coarse skewed (-1.30 to -0.43), symmetrical (-0.43 to 0.43), fine skewed (0.43 to 1.30), very fine skewed (>1.30)",
            "Very platykurtic (<1.70), platykurtic (1.70 to 2.55), mesokurtic (2.55 to 3.70), leptokurtic (3.70 to 7.40), very leptokurtic (>7.40)",
        ],
        "Formula": [
            "sum(percent * phi) / 100",
            "sqrt(sum(percent * (phi - mean)^2) / 100)",
            "sum(percent * (phi - mean)^3) / (100 * standard_deviation^3)",
            "sum(percent * (phi - mean)^4) / (100 * standard_deviation^4)",
        ],
    })

    return {
        "phi_scale": phi_scale,
        "graphical_criteria": graphical_criteria,
        "moments_criteria": moments_criteria,
    }


def build_sta_input(input_df: pd.DataFrame, results_df: pd.DataFrame, sample_col: str, method: str) -> Optional[pd.DataFrame]:
    """Build STA input table when x and y columns are available."""
    x_col = find_any_column(input_df, ["x", "easting", "utm_x", "x_etrs89", "x (etrs89_portugal_tm06)"])
    y_col = find_any_column(input_df, ["y", "northing", "utm_y", "y_etrs89", "y (etrs89_portugal_tm06)"])
    if x_col is None or y_col is None:
        return None

    coords = input_df[[sample_col, x_col, y_col]].copy()
    coords.columns = ["id", "x", "y"]
    if method == "graphical":
        stats = results_df[["sample", "mean_size_phi", "sorting", "skewness"]].copy()
        stats.columns = ["id", "Mz", "Sig", "Sk"]
    else:
        stats = results_df[["sample", "mean_size_phi", "standard_deviation", "skewness"]].copy()
        stats.columns = ["id", "Mz", "Sig", "Sk"]

    coords["id"] = coords["id"].astype(str).str.strip()
    stats["id"] = stats["id"].astype(str).str.strip()
    out = coords.merge(stats, on="id", how="inner")

    for col in ["x", "y", "Mz", "Sig", "Sk"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["id", "x", "y", "Mz", "Sig", "Sk"]).reset_index(drop=True)
    return out if not out.empty else None


def build_graphical_qa_row(sample_name: str, unit_used: str, result: dict, cfg: Config) -> dict:
    """Build one graphical-method QA diagnostics row."""
    percentile_cols = [f"phi{p}" for p in PERCENTILES]
    total = float(result.get("total_raw", np.nan))
    issues = []

    if total <= 0 or result.get("_cum_raw") is None:
        raw_monotonic = False
        issues.append("No material or missing cumulative curve.")
    else:
        cum_raw = np.asarray(result["_cum_raw"], dtype=float)
        diffs = np.diff(cum_raw[np.isfinite(cum_raw)])
        raw_monotonic = bool(np.all(diffs >= -float(cfg.monotonicity_tolerance)))
        if not raw_monotonic:
            issues.append("Raw cumulative curve is not monotonic.")

    values = {col: result.get(col, np.nan) for col in percentile_cols}
    has_nan = bool(np.isnan([values[col] for col in percentile_cols]).any())
    if has_nan:
        issues.append("One or more percentiles are NaN.")

    phi5 = values["phi5"]
    phi16 = values["phi16"]
    phi25 = values["phi25"]
    phi50 = values["phi50"]
    phi75 = values["phi75"]
    phi84 = values["phi84"]
    phi95 = values["phi95"]
    center_zero = bool(np.isfinite(phi84) and np.isfinite(phi16) and (phi84 - phi16) == 0)
    tail_zero = bool(np.isfinite(phi95) and np.isfinite(phi5) and (phi95 - phi5) == 0)
    iqr_zero = bool(np.isfinite(phi75) and np.isfinite(phi25) and (phi75 - phi25) == 0)

    if center_zero:
        issues.append("phi84 - phi16 = 0.")
    if tail_zero:
        issues.append("phi95 - phi5 = 0.")
    if iqr_zero:
        issues.append("phi75 - phi25 = 0; kurtosis undefined.")
    if not np.isfinite(phi50):
        issues.append("phi50 is undefined.")
    else:
        if np.isfinite(phi16) and phi50 < phi16:
            issues.append("phi50 < phi16.")
        if np.isfinite(phi84) and phi50 > phi84:
            issues.append("phi50 > phi84.")

    return {
        "method": "graphical",
        "sample": sample_name,
        "units_used": unit_used,
        "total_raw_sum": total,
        "raw_cumulative_monotonic": raw_monotonic,
        "has_nan_percentiles": has_nan,
        "center_width_zero": center_zero,
        "tail_width_zero": tail_zero,
        "iqr_zero": iqr_zero,
        "issues": "; ".join(issues) if issues else "OK",
    }


def build_moments_qa_row(sample_name: str, unit_used: str, result: dict) -> dict:
    """Build one moments-method QA diagnostics row."""
    total = float(result.get("total_raw", np.nan))
    sd = float(result.get("standard_deviation", np.nan))
    variance = float(result.get("variance", np.nan))
    issues = []

    if total <= 0:
        issues.append("No material.")
    if not np.isfinite(variance):
        issues.append("Variance is undefined.")
    elif variance < 0:
        issues.append("Variance is negative.")
    if not np.isfinite(sd):
        issues.append("Standard deviation is undefined.")
    elif sd == 0:
        issues.append("Standard deviation is zero; skewness and kurtosis are undefined.")

    return {
        "method": "moments",
        "sample": sample_name,
        "units_used": unit_used,
        "total_raw_sum": total,
        "variance": variance,
        "standard_deviation_zero": bool(np.isfinite(sd) and sd == 0),
        "has_nan_statistics": bool(
            np.isnan([
                result.get("mean_size_phi", np.nan),
                result.get("standard_deviation", np.nan),
                result.get("skewness", np.nan),
                result.get("kurtosis", np.nan),
            ]).any()
        ),
        "issues": "; ".join(issues) if issues else "OK",
    }


def summarize_numeric(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row summary table containing numeric means."""
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    return pd.DataFrame([{
        "sample": "MEAN",
        **{col: results_df[col].mean() for col in numeric_cols},
    }])


def build_plot_data_tables(
    phi_values: np.ndarray,
    data: pd.DataFrame,
    sample_names: list[str],
) -> dict[str, pd.DataFrame]:
    """Build normalized frequency and cumulative tables for external plotting."""
    long_rows = []
    frequency_wide = pd.DataFrame({"sample": sample_names})
    cumulative_wide = pd.DataFrame({"sample": sample_names})

    phi_values = np.asarray(phi_values, dtype=float)
    phi_columns = [f"phi_{phi:.6g}" for phi in phi_values]
    frequency_values = []
    cumulative_values = []

    for sample_name, row in zip(sample_names, data.itertuples(index=False)):
        freq_percent, total = normalize_frequencies(pd.Series(row, index=data.columns))
        cumulative = freq_percent.cumsum()

        frequency_values.append(freq_percent.to_numpy(dtype=float))
        cumulative_values.append(cumulative.to_numpy(dtype=float))

        for grain_col, phi, freq, cum in zip(data.columns, phi_values, freq_percent, cumulative):
            long_rows.append({
                "sample": sample_name,
                "grain_size_column": grain_col,
                "phi": float(phi),
                "grain_size_mm": float(phi_to_mm(phi)),
                "frequency_percent": float(freq),
                "cumulative_percent": float(cum),
                "total_raw_sum": float(total),
            })

    if frequency_values:
        for i, column in enumerate(phi_columns):
            frequency_wide[column] = [row[i] for row in frequency_values]
            cumulative_wide[column] = [row[i] for row in cumulative_values]

    phi_reference = pd.DataFrame({
        "phi": phi_values,
        "grain_size_mm": phi_to_mm(phi_values),
        "wide_column": phi_columns,
        "source_grain_size_column": list(data.columns),
    })

    return {
        "Plot_Data_Long": pd.DataFrame(long_rows),
        "Plot_Frequency_Wide": frequency_wide,
        "Plot_Cumulative_Wide": cumulative_wide,
        "Plot_Phi_Reference": phi_reference,
    }


def process_graphical(
    phi_values: np.ndarray,
    data: pd.DataFrame,
    sample_names: list[str],
    unit_used: str,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the Folk and Ward graphical method."""
    analyzer = FolkWardAnalyzer(cumulative_max_percent=cfg.cumulative_max_percent)
    results = []
    qa_rows = []
    percentile_cols = [f"phi{p}" for p in PERCENTILES]

    for sample_name, row in zip(sample_names, data.itertuples(index=False)):
        freqs = pd.Series(row, index=data.columns)
        result = analyzer.analyze_sample(phi_values, freqs)
        results.append({
            "sample": sample_name,
            **{col: result.get(col, np.nan) for col in percentile_cols},
            "mean_size_phi": result["mean_size_phi"],
            "mean_size_mm": result["mean_size_mm"],
            "sorting": result["sorting"],
            "skewness": result["skewness"],
            "kurtosis": result["kurtosis"],
            "grain_size_class": result["grain_size_class"],
            "sorting_class": result["sorting_class"],
            "skewness_class": result["skewness_class"],
            "kurtosis_class": result["kurtosis_class"],
        })
        qa_rows.append(build_graphical_qa_row(sample_name, unit_used, result, cfg))

    results_df = pd.DataFrame(results)
    percentiles_mm = results_df[["sample"]].copy()
    for col in percentile_cols:
        percentiles_mm[col.replace("phi", "d_mm_at_")] = phi_to_mm(results_df[col])
    return results_df, percentiles_mm, pd.DataFrame(qa_rows)


def process_moments(
    phi_values: np.ndarray,
    data: pd.DataFrame,
    sample_names: list[str],
    unit_used: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the moments method."""
    analyzer = MomentsAnalyzer()
    results = []
    qa_rows = []

    for sample_name, row in zip(sample_names, data.itertuples(index=False)):
        freqs = pd.Series(row, index=data.columns)
        result = analyzer.analyze_sample(phi_values, freqs)
        results.append({
            "sample": sample_name,
            "mean_size_phi": result["mean_size_phi"],
            "mean_size_mm": result["mean_size_mm"],
            "variance": result["variance"],
            "standard_deviation": result["standard_deviation"],
            "skewness": result["skewness"],
            "kurtosis": result["kurtosis"],
            "grain_size_class": result["grain_size_class"],
            "standard_deviation_class": result["standard_deviation_class"],
            "skewness_class": result["skewness_class"],
            "kurtosis_class": result["kurtosis_class"],
        })
        qa_rows.append(build_moments_qa_row(sample_name, unit_used, result))

    return pd.DataFrame(results), pd.DataFrame(qa_rows)


def build_comparison_sheet(graphical_df: pd.DataFrame, moments_df: pd.DataFrame) -> pd.DataFrame:
    """Compare graphical and moment statistics sample by sample."""
    g = graphical_df[["sample", "mean_size_phi", "sorting", "skewness", "kurtosis"]].copy()
    m = moments_df[["sample", "mean_size_phi", "standard_deviation", "skewness", "kurtosis"]].copy()
    g.columns = ["sample", "graphical_mean_phi", "graphical_sorting", "graphical_skewness", "graphical_kurtosis"]
    m.columns = ["sample", "moments_mean_phi", "moments_standard_deviation", "moments_skewness", "moments_kurtosis"]
    out = g.merge(m, on="sample", how="outer")
    out["delta_mean_phi_graphical_minus_moments"] = out["graphical_mean_phi"] - out["moments_mean_phi"]
    out["delta_sorting_graphical_minus_moments_sd"] = out["graphical_sorting"] - out["moments_standard_deviation"]
    out["delta_skewness_graphical_minus_moments"] = out["graphical_skewness"] - out["moments_skewness"]
    out["delta_kurtosis_graphical_minus_moments"] = out["graphical_kurtosis"] - out["moments_kurtosis"]
    return out


def process_table(input_df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame | str]:
    """Process one sediment texture table and return output sheets."""
    df = normalize_columns(input_df)
    grain_cols, size_raw = parse_grain_size_columns(df)
    sample_col = choose_sample_column(df, grain_cols, cfg.sample_column)

    if cfg.units == "auto":
        is_mm = looks_like_mm(size_raw)
    else:
        is_mm = cfg.units == "mm"

    if is_mm:
        size_phi = mm_to_phi(size_raw)
        unit_used = "mm converted to phi"
    else:
        size_phi = size_raw
        unit_used = "phi"

    order = np.argsort(size_phi)
    phi_values = size_phi[order]
    ordered_grain_cols = [grain_cols[i] for i in order]
    data = df[ordered_grain_cols].copy()
    sample_names = df[sample_col].astype(str).tolist()

    sheets: dict[str, pd.DataFrame | str] = {}
    qa_tables = []
    refs = create_reference_tables()

    if cfg.method in {"graphical", "both"}:
        graphical_df, percentiles_mm, graphical_qa = process_graphical(phi_values, data, sample_names, unit_used, cfg)
        sheets["Graphical_Parameters_phi"] = graphical_df
        sheets["Graphical_Parameters_phi_MEAN"] = summarize_numeric(graphical_df)
        sheets["Graphical_Percentiles_mm"] = percentiles_mm
        sheets["Graphical_Percentiles_mm_MEAN"] = summarize_numeric(percentiles_mm)
        qa_tables.append(graphical_qa)

        sta_df = build_sta_input(df, graphical_df, sample_col, method="graphical")
        if sta_df is not None:
            sheets["STA_Input_Graphical"] = sta_df

    if cfg.method in {"moments", "both"}:
        moments_df, moments_qa = process_moments(phi_values, data, sample_names, unit_used)
        sheets["Moments_Parameters_phi"] = moments_df
        sheets["Moments_Parameters_phi_MEAN"] = summarize_numeric(moments_df)
        qa_tables.append(moments_qa)

        sta_df = build_sta_input(df, moments_df, sample_col, method="moments")
        if sta_df is not None:
            sheets["STA_Input_Moments"] = sta_df

    if cfg.method == "both":
        sheets["Graphical_vs_Moments"] = build_comparison_sheet(
            sheets["Graphical_Parameters_phi"], sheets["Moments_Parameters_phi"]
        )

    metadata = pd.DataFrame({
        "parameter": [
            "routine",
            "method",
            "input_units_mode",
            "units_used",
            "sample_column",
            "grain_size_columns_used",
            "cumulative_max_percent",
            "monotonicity_tolerance",
        ],
        "value": [
            "sediment_textural_analysis_cli",
            cfg.method,
            cfg.units,
            unit_used,
            sample_col,
            len(ordered_grain_cols),
            cfg.cumulative_max_percent,
            cfg.monotonicity_tolerance,
        ],
    })

    sheets.update(build_plot_data_tables(phi_values, data, sample_names))
    sheets["QA"] = pd.concat(qa_tables, ignore_index=True) if qa_tables else pd.DataFrame()
    sheets["Phi_Scale_Reference"] = refs["phi_scale"]
    sheets["Graphical_Classification_Criteria"] = refs["graphical_criteria"]
    sheets["Moments_Classification_Criteria"] = refs["moments_criteria"]
    sheets["metadata"] = metadata
    sheets["_unit_used"] = unit_used
    sheets["_plot_context"] = {
        "phi_values": phi_values,
        "data": data,
        "sample_names": sample_names,
    }
    return sheets


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


def setup_probability_axis(
    ax,
    phi_values: np.ndarray,
    min_percent: float = 0.02,
    max_percent: float = 99.98,
) -> None:
    """Apply probability-paper axes, ticks, grids, and labels."""
    major_percent, minor_percent = probability_percent_ticks()
    major_probit = probit_from_percent(major_percent)
    minor_probit = probit_from_percent(minor_percent)

    ax.set_ylim(probit_from_percent([min_percent, max_percent]))
    ax.set_yticks(major_probit)
    ax.set_yticklabels([f"{y:.2f}" for y in major_percent], fontsize=8, fontweight="bold")
    ax.set_yticks(minor_probit, minor=True)

    finite_phi = np.asarray(phi_values, dtype=float)
    finite_phi = finite_phi[np.isfinite(finite_phi)]
    if finite_phi.size:
        x_min = float(np.floor(finite_phi.min() * 2.0) / 2.0)
        x_max = float(np.ceil(finite_phi.max() * 2.0) / 2.0)
    else:
        x_min, x_max = -5.0, 5.0
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5

    major_x = np.arange(x_min, x_max + 0.5, 0.5)
    minor_x = np.arange(x_min, x_max + 0.1, 0.1)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(major_x)
    ax.set_xticks(minor_x, minor=True)
    ax.set_xticklabels([f"{x:.1f}" for x in major_x], fontsize=10, fontweight="bold")

    ax.set_xlabel("Grain Size (phi units)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Cumulative Percentage (%)", fontsize=10, fontweight="bold")
    ax.grid(True, which="major", linestyle="-", color="black", linewidth=1.0)
    ax.grid(True, which="minor", linestyle="--", color="gray", linewidth=0.5)

    ax_top = ax.twiny()
    ax_top.set_xlim(x_min, x_max)
    top_ticks = np.arange(np.ceil(x_min), np.floor(x_max) + 1.0, 1.0)
    ax_top.set_xticks(top_ticks)
    ax_top.set_xticklabels([f"{phi_to_mm(phi):.3f}" for phi in top_ticks], fontsize=8, fontweight="bold")
    ax_top.set_xlabel("Grain Size (mm)", fontsize=10, fontweight="bold")


def generate_probability_plots(sheets: dict, output_path: Path, cfg: Config) -> list[Path]:
    """Generate probability-paper and cumulative-curve graphical outputs."""
    if not cfg.plot_probability_paper and not cfg.plot_cumulative_curves:
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Plot output skipped: matplotlib is required for probability-paper plots ({exc}).")
        return []

    context = sheets.get("_plot_context")
    if not isinstance(context, dict):
        return []

    phi_values = np.asarray(context["phi_values"], dtype=float)
    data = context["data"]
    sample_names = context["sample_names"]
    plot_dir = cfg.plot_output_dir or output_path.with_suffix("").with_name(f"{output_path.stem}_figures")
    plot_dir.mkdir(parents=True, exist_ok=True)
    fmt = cfg.plot_format.lower().lstrip(".")
    dpi = 600 if fmt in {"jpg", "jpeg", "png", "tif", "tiff"} else 300
    saved: list[Path] = []
    plot_min_percent = 0.02
    plot_max_percent = min(float(cfg.cumulative_max_percent), 99.98)

    fig_size_inches = (297 / 25.4, 210 / 25.4)

    if cfg.plot_probability_paper:
        fig, ax = plt.subplots(figsize=fig_size_inches)
        setup_probability_axis(ax, phi_values, min_percent=plot_min_percent, max_percent=plot_max_percent)
        ax.set_title("Probability Paper", fontsize=11, fontweight="bold")
        out = plot_dir / f"PROBABILITY_PAPER.{fmt}"
        fig.tight_layout()
        fig.savefig(out, dpi=dpi, format=fmt, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)

    if cfg.plot_cumulative_curves:
        fig, ax = plt.subplots(figsize=fig_size_inches)
        setup_probability_axis(ax, phi_values, min_percent=plot_min_percent, max_percent=plot_max_percent)
        colors = plt.colormaps["tab20"]

        for i, (sample_name, row) in enumerate(zip(sample_names, data.itertuples(index=False))):
            freq_percent, total = normalize_frequencies(pd.Series(row, index=data.columns))
            if total <= 0:
                continue
            cumulative = freq_percent.cumsum().to_numpy(dtype=float)
            cumulative_plot = np.minimum(cumulative, plot_max_percent)
            valid = (cumulative_plot >= plot_min_percent) & (cumulative_plot <= plot_max_percent)
            if not np.any(valid):
                continue
            ax.plot(
                phi_values[valid],
                probit_from_percent(cumulative_plot[valid]),
                marker="o",
                linestyle="-",
                markersize=5,
                label=str(sample_name),
                color=colors(i % 20),
            )

        ax.set_title("Grain Size Distribution", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
        out = plot_dir / f"PROBABILITY_PLOT.{fmt}"
        fig.tight_layout()
        fig.savefig(out, dpi=dpi, format=fmt, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)

    return saved


def write_output_workbook(
    output_path: Path,
    sheets: dict,
    sta_output_path: Optional[Path] = None,
) -> None:
    """Write the output Excel workbook."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, value in sheets.items():
            if name.startswith("_") or not isinstance(value, pd.DataFrame):
                continue
            if name.endswith("_MEAN"):
                continue

            value.to_excel(writer, sheet_name=name[:31], index=False)
            mean_name = f"{name}_MEAN"
            if mean_name in sheets:
                sheets[mean_name].to_excel(
                    writer,
                    sheet_name=name[:31],
                    startrow=len(value) + 1,
                    header=False,
                    index=False,
                )

    if sta_output_path is not None:
        sta_sheets = {name: value for name, value in sheets.items() if name.startswith("STA_Input_")}
        if sta_sheets:
            sta_output_path.parent.mkdir(parents=True, exist_ok=True)
            with pd.ExcelWriter(sta_output_path, engine="openpyxl") as writer:
                for name, value in sta_sheets.items():
                    value.to_excel(writer, sheet_name=name[:31], index=False)


def process_file(
    input_path: Path,
    output_path: Path,
    cfg: Config,
    sta_output_path: Optional[Path] = None,
) -> Path:
    """Run sediment textural analysis for one input file."""
    print("=== Sediment textural analysis ===")
    print(f"Method: {cfg.method}")
    print(f"Input: {input_path}")

    df = read_input_table(input_path, cfg.sheet_name)
    sheets = process_table(df, cfg)
    write_output_workbook(output_path, sheets, sta_output_path=sta_output_path)
    plot_paths = generate_probability_plots(sheets, output_path, cfg)

    print(f"Output saved: {output_path}")
    for plot_path in plot_paths:
        print(f"Plot saved: {plot_path}")
    if any(name.startswith("STA_Input_") for name in sheets):
        print("STA_Input sheet generated.")
        if sta_output_path is not None:
            print(f"Dedicated STA workbook saved: {sta_output_path}")
    else:
        print("STA_Input was not generated because x/y coordinate columns were not found.")
    print("Processing completed successfully.")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate sediment textural parameters from grain-size frequency data "
            "using the Folk and Ward graphical method, the moments method, or both."
        )
    )
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT_PATH, type=Path, help="Input CSV or Excel file.")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_PATH, help="Output Excel workbook.")
    parser.add_argument("--method", choices=["graphical", "moments", "both"], default="both", help="Analysis method.")
    parser.add_argument("--sheet", default=None, help="Excel sheet name. Ignored for CSV input.")
    parser.add_argument("--units", choices=["auto", "phi", "mm"], default="auto", help="Units of grain-size column headers.")
    parser.add_argument("--sample-column", default=None, help="Sample identifier column. Default: first non-grain-size column.")
    parser.add_argument("--sta-output", default=None, help="Optional dedicated STA_Input Excel workbook.")
    parser.add_argument(
        "--plot-probability-paper",
        action="store_true",
        help="Export a blank probability-paper plot with phi and mm axes.",
    )
    parser.add_argument(
        "--plot-cumulative-curves",
        action="store_true",
        help="Export cumulative sample curves on probability paper.",
    )
    parser.add_argument(
        "--plot-output-dir",
        type=Path,
        default=None,
        help="Directory for probability-paper figures. Default: output workbook stem plus '_figures'.",
    )
    parser.add_argument(
        "--plot-format",
        choices=["jpg", "jpeg", "png", "pdf", "svg", "tif", "tiff"],
        default="jpg",
        help="Figure format for probability-paper outputs.",
    )
    parser.add_argument(
        "--cumulative-max-percent",
        type=float,
        default=99.98,
        help="Maximum cumulative percent used for graphical-method interpolation.",
    )
    parser.add_argument("--monotonicity-tolerance", type=float, default=1e-8, help="QA tolerance for cumulative monotonicity.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point."""
    args = build_parser().parse_args(argv)
    cfg = Config(
        method=args.method,
        units=args.units,
        sheet_name=args.sheet,
        sample_column=args.sample_column,
        cumulative_max_percent=args.cumulative_max_percent,
        monotonicity_tolerance=args.monotonicity_tolerance,
        plot_probability_paper=args.plot_probability_paper,
        plot_cumulative_curves=args.plot_cumulative_curves,
        plot_output_dir=args.plot_output_dir,
        plot_format=args.plot_format,
    )
    output_path = default_output_path(args.input, args.output)
    sta_output_path = optional_path(args.sta_output)
    process_file(args.input, output_path, cfg, sta_output_path=sta_output_path)
    return 0


if __name__ == "__main__":
    main()
