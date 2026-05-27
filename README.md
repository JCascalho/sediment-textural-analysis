# Sediment Textural Analysis

Python routine for calculating sediment textural parameters from grain-size
frequency distributions using the Folk and Ward (1957) graphical method.

The routine calculates:

- grain-size percentiles in phi units;
- percentile equivalents in millimetres;
- mean grain size;
- inclusive graphic sorting;
- inclusive graphic skewness;
- graphic kurtosis;
- textural classifications;
- quality-control diagnostics;
- optional `STA_Input` table when `x` and `y` coordinates are available.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Input Format

The input file can be CSV or Excel.

The table must contain:

- one sample identifier column;
- grain-size class columns with numeric headers;
- frequency values in each sample row.

Example with phi headers:

```text
sample_id,x,y,-1,0,1,2,3,4
S01,100,200,0,5,20,45,25,5
S02,150,220,0,2,18,50,27,3
```

Example with millimetre headers:

```text
sample_id,x,y,2,1,0.5,0.25,0.125,0.063
S01,100,200,0,5,20,45,25,5
S02,150,220,0,2,18,50,27,3
```

The frequency rows do not need to sum exactly to 100; they are normalized
internally.

## Usage

CSV input:

```bash
python sediment_textural_analysis.py --input examples/example_textural_input_phi.csv --output results/textural_results.xlsx --units phi
```

Excel input:

```bash
python sediment_textural_analysis.py --input grain_size_data.xlsx --sheet Sheet1 --output textural_results.xlsx --units auto
```

Create a dedicated STA bridge workbook:

```bash
python sediment_textural_analysis.py --input grain_size_data.xlsx --output textural_results.xlsx --sta-output sta_input.xlsx
```

## Outputs

The output Excel workbook includes:

- `Textural_Parameters_phi`
- `Percentiles_mm`
- `QA`
- `STA_Input`, when coordinate columns are available
- `Phi_Scale_Reference`
- `Classification_Criteria`
- `metadata`

## Notes

This routine uses the Folk and Ward (1957) graphical method. Grain-size classes
are internally ordered from coarser to finer in phi space. If units are set to
`auto`, numeric headers that look like positive millimetre classes are converted
to phi units.
