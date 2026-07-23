# Sediment Textural Analysis

Python routines for educational sediment textural analysis using Folk and Ward graphical parameters and method-of-moments statistics.

This repository provides a reproducible, step-by-step workflow for processing grain-size frequency distributions from an educational coastal sediment dataset. The routines are intentionally kept separate for pedagogical reasons:

- `sediment_textural_graphical_moments_parameters.py` calculates sediment textural parameters.
- `sediment_textural_graphical_moments_plots.py` generates publication-style plots from the output workbook.

## EDUCOAST Dataset

The included dataset, `Data_base_EDUCOAST.xlsx`, is an educational grain-size database associated with the EDUCOAST project, Nature-Based Education in Coastal Geosciences.

As part of the EDUCOAST project, field surveys were conducted in an area encompassing the salt marshes of the Ria Formosa lagoon and Cabanas Barrier Island, in the Tavira region of southern Portugal. During these educational field activities, sediment samples were collected from a range of coastal and back-barrier environments, including the Atlantic beach, barrier-island interior, tidal-channel margins, and salt-marsh areas.

## Workflow

```text
Data_base_EDUCOAST.xlsx
        |
        v
sediment_textural_graphical_moments_parameters.py
        |
        v
output_textural_parameters_Data_base_EDUCOAST.xlsx
        |
        v
sediment_textural_graphical_moments_plots.py
        |
        v
figures/
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Parameter Routine

```bash
python sediment_textural_graphical_moments_parameters.py
```

By default, this reads:

```text
Data_base_EDUCOAST.xlsx
```

and writes:

```text
output_textural_parameters_Data_base_EDUCOAST.xlsx
```

## Run the Plotting Routine

```bash
python sediment_textural_graphical_moments_plots.py
```

By default, this reads:

```text
output_textural_parameters_Data_base_EDUCOAST.xlsx
```

and writes figures to:

```text
figures/
```

## Explicit Usage

```bash
python sediment_textural_graphical_moments_parameters.py --input Data_base_EDUCOAST.xlsx --output output_textural_parameters_Data_base_EDUCOAST.xlsx --units auto --method both
python sediment_textural_graphical_moments_plots.py --workbook output_textural_parameters_Data_base_EDUCOAST.xlsx --output-dir figures --plot all
```

## Citation

If you use these routines or the EDUCOAST educational dataset, cite the archived Zenodo release corresponding to the exact version used.

## References

Folk, R.L., and Ward, W.C., 1957. Brazos River bar: a study in the significance of grain size parameters. Journal of Sedimentary Petrology, 27, 3-26.

Friedman, G.M., 1962. On sorting, sorting coefficients, and the lognormality of the grain-size distribution of sandstones. Journal of Geology, 70, 737-753.
