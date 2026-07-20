"""Convert CSV files to XLSX workbooks.

This utility converts either one CSV file or all CSV files in a folder to
Excel XLSX format.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def convert_csv_to_xlsx(input_path: Path, output_path: Path | None = None) -> list[Path]:
    """Convert one CSV file or all CSV files in a folder to XLSX."""
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"Input file is not a CSV file: {input_path}")

        if output_path is None:
            output_path = input_path.with_suffix(".xlsx")
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(input_path)
        df.to_excel(output_path, index=False)
        return [output_path]

    if input_path.is_dir():
        if output_path is None:
            output_dir = input_path
        else:
            output_dir = Path(output_path)

        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        for csv_file in sorted(input_path.glob("*.csv")):
            xlsx_file = output_dir / f"{csv_file.stem}.xlsx"
            df = pd.read_csv(csv_file)
            df.to_excel(xlsx_file, index=False)
            saved.append(xlsx_file)

        return saved

    raise FileNotFoundError(f"Input path not found: {input_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert CSV file(s) to XLSX format.")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Input CSV file or folder containing CSV files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        type=Path,
        help="Output XLSX file or output folder. Default: same location as input.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    saved = convert_csv_to_xlsx(args.input, args.output)

    for path in saved:
        print(f"Saved: {path}")

    if not saved:
        print("No CSV files found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
