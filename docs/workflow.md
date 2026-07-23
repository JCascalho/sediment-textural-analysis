# Workflow

The repository is organized as a two-stage educational workflow.

## Stage 1 - Textural Parameters

Run:

```bash
python sediment_textural_graphical_moments_parameters.py
```

Input:

```text
Data_base_EDUCOAST.xlsx
```

Output:

```text
output_textural_parameters_Data_base_EDUCOAST.xlsx
```

## Stage 2 - Plots

Run:

```bash
python sediment_textural_graphical_moments_plots.py
```

Input:

```text
output_textural_parameters_Data_base_EDUCOAST.xlsx
```

Output:

```text
figures/
```

The routines are kept separate so users can inspect the calculation stage before generating graphical outputs.
