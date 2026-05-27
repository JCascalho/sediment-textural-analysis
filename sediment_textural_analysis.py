"""Sediment textural analysis using the Folk and Ward graphical method.

This command-line routine calculates percentile-based sediment textural
parameters from grain-size frequency distributions:

- mean grain size;
- inclusive graphic sorting;
- inclusive graphic skewness;
- graphic kurtosis;
- textural classifications;
- quality-control diagnostics;
- optional STA-ready export when x/y coordinates are present.

Input grain-size classes can be supplied as phi or millimetres in the column
headers. Frequencies are normalized internally, so rows may sum to 100, 1, or
any positive total.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


INPUT_PREFIX = "INPUT_"
OUTPUT_PREFIX = "OUTPUT_"
PERCENTILES = (5, 16, 25, 50, 75, 84, 95)


@dataclass
class Config:
    """Runtime settings for sediment textural analysis."""

    units: str = "auto"
    sheet_name: Optional[str] = None
    sample_column: Optional[str] = None
    cumulative_max_percent: float = 99.98
    monotonicity_tolerance: float = 1e-8


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
        return "Gravel"

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
        freq = pd.to_numeric(frequencies, errors="coerce").fillna(0.0).clip(lower=0.0)
        total = float(freq.sum())

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

        cumulative_raw = (freq.cumsum() / total) * 100.0
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

    classification = pd.DataFrame({
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

    return {
        "phi_scale": phi_scale,
        "classification_criteria": classification,
    }


def build_sta_input(input_df: pd.DataFrame, results_df: pd.DataFrame, sample_col: str) -> Optional[pd.DataFrame]:
    """Build STA input table when x and y columns are available."""
    x_col = find_any_column(input_df, ["x", "easting", "utm_x", "x_etrs89", "x (etrs89_portugal_tm06)"])
    y_col = find_any_column(input_df, ["y", "northing", "utm_y", "y_etrs89", "y (etrs89_portugal_tm06)"])
    if x_col is None or y_col is None:
        return None

    coords = input_df[[sample_col, x_col, y_col]].copy()
    coords.columns = ["id", "x", "y"]
    stats = results_df[["sample", "mean_size_phi", "sorting", "skewness"]].copy()
    stats.columns = ["id", "Mz", "Sig", "Sk"]

    coords["id"] = coords["id"].astype(str).str.strip()
    stats["id"] = stats["id"].astype(str).str.strip()
    out = coords.merge(stats, on="id", how="inner")

    for col in ["x", "y", "Mz", "Sig", "Sk"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["id", "x", "y", "Mz", "Sig", "Sk"]).reset_index(drop=True)
    return out if not out.empty else None


def build_qa_row(
    sample_name: str,
    unit_used: str,
    result: dict,
    cfg: Config,
) -> dict:
    """Build one QA diagnostics row."""
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

    analyzer = FolkWardAnalyzer(cumulative_max_percent=cfg.cumulative_max_percent)
    results = []
    qa_rows = []
    percentile_cols = [f"phi{p}" for p in PERCENTILES]

    for sample_name, row in zip(sample_names, data.itertuples(index=False)):
        freqs = pd.Series(row, index=ordered_grain_cols)
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
        qa_rows.append(build_qa_row(sample_name, unit_used, result, cfg))

    results_df = pd.DataFrame(results)
    percentiles_mm = results_df[["sample"]].copy()
    for col in percentile_cols:
        percentiles_mm[col.replace("phi", "d_mm_at_")] = phi_to_mm(results_df[col])

    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    summary_phi = pd.DataFrame([{
        "sample": "MEAN",
        **{col: results_df[col].mean() for col in numeric_cols},
    }])
    summary_mm = pd.DataFrame([{
        "sample": "MEAN",
        **{col: percentiles_mm[col].mean() for col in percentiles_mm.columns if col != "sample"},
    }])

    qa_df = pd.DataFrame(qa_rows)
    refs = create_reference_tables()
    sta_df = build_sta_input(df, results_df, sample_col)

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
            "sediment_textural_analysis",
            "Folk and Ward (1957) graphical method",
            cfg.units,
            unit_used,
            sample_col,
            len(ordered_grain_cols),
            cfg.cumulative_max_percent,
            cfg.monotonicity_tolerance,
        ],
    })

    sheets: dict[str, pd.DataFrame | str] = {
        "Textural_Parameters_phi": results_df,
        "Textural_Parameters_phi_MEAN": summary_phi,
        "Percentiles_mm": percentiles_mm,
        "Percentiles_mm_MEAN": summary_mm,
        "QA": qa_df,
        "Phi_Scale_Reference": refs["phi_scale"],
        "Classification_Criteria": refs["classification_criteria"],
        "metadata": metadata,
        "_unit_used": unit_used,
    }
    if sta_df is not None:
        sheets["STA_Input"] = sta_df
    return sheets


def write_output_workbook(
    output_path: Path,
    sheets: dict[str, pd.DataFrame | str],
    sta_output_path: Optional[Path] = None,
) -> None:
    """Write the output Excel workbook."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        results = sheets["Textural_Parameters_phi"]
        results.to_excel(writer, sheet_name="Textural_Parameters_phi", index=False)
        sheets["Textural_Parameters_phi_MEAN"].to_excel(
            writer,
            sheet_name="Textural_Parameters_phi",
            startrow=len(results) + 1,
            header=False,
            index=False,
        )

        percentiles_mm = sheets["Percentiles_mm"]
        percentiles_mm.to_excel(writer, sheet_name="Percentiles_mm", index=False)
        sheets["Percentiles_mm_MEAN"].to_excel(
            writer,
            sheet_name="Percentiles_mm",
            startrow=len(percentiles_mm) + 1,
            header=False,
            index=False,
        )

        for name in ["QA", "STA_Input", "Phi_Scale_Reference", "Classification_Criteria", "metadata"]:
            if name in sheets:
                sheets[name].to_excel(writer, sheet_name=name, index=False)

    if sta_output_path is not None and "STA_Input" in sheets:
        sta_output_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(sta_output_path, engine="openpyxl") as writer:
            sheets["STA_Input"].to_excel(writer, sheet_name="STA_Input", index=False)


def process_file(
    input_path: Path,
    output_path: Path,
    cfg: Config,
    sta_output_path: Optional[Path] = None,
) -> Path:
    """Run sediment textural analysis for one input file."""
    print("=== Sediment textural analysis: Folk and Ward graphical method ===")
    print(f"Input: {input_path}")

    df = read_input_table(input_path, cfg.sheet_name)
    sheets = process_table(df, cfg)
    write_output_workbook(output_path, sheets, sta_output_path=sta_output_path)

    print(f"Output saved: {output_path}")
    if "STA_Input" in sheets:
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
        description="Calculate Folk and Ward sediment textural parameters from grain-size frequency data."
    )
    parser.add_argument("--input", "-i", required=True, type=Path, help="Input CSV or Excel file.")
    parser.add_argument("--output", "-o", default=None, help="Output Excel workbook.")
    parser.add_argument("--sheet", default=None, help="Excel sheet name. Ignored for CSV input.")
    parser.add_argument("--units", choices=["auto", "phi", "mm"], default="auto", help="Units of grain-size column headers.")
    parser.add_argument("--sample-column", default=None, help="Sample identifier column. Default: first non-grain-size column.")
    parser.add_argument("--sta-output", default=None, help="Optional dedicated STA_Input Excel workbook.")
    parser.add_argument("--cumulative-max-percent", type=float, default=99.98, help="Maximum cumulative percent used for interpolation.")
    parser.add_argument("--monotonicity-tolerance", type=float, default=1e-8, help="QA tolerance for cumulative monotonicity.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point."""
    args = build_parser().parse_args(argv)
    cfg = Config(
        units=args.units,
        sheet_name=args.sheet,
        sample_column=args.sample_column,
        cumulative_max_percent=args.cumulative_max_percent,
        monotonicity_tolerance=args.monotonicity_tolerance,
    )
    output_path = default_output_path(args.input, args.output)
    sta_output_path = optional_path(args.sta_output)
    process_file(args.input, output_path, cfg, sta_output_path=sta_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
