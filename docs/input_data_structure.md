# Input Data Structure

The EDUCOAST workbook, `Data_base_EDUCOAST.xlsx`, contains sediment grain-size frequency data organized as a table.

Expected structure:

- one row per sediment sample;
- one sample identifier column;
- optional `x` and `y` coordinate columns;
- numeric grain-size class columns;
- frequency values in each sample row.

Grain-size class headers may be expressed in phi or millimetres. Use `--units auto` unless the table is ambiguous.

Rows do not need to sum exactly to 100 because the parameter routine normalizes frequency totals internally.
