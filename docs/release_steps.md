# GitHub and Zenodo Release Steps

## 1. Update GitHub

Copy the files in this update package into the existing repository:

```text
JCascalho/sediment-textural-analysis
```

Recommended version:

```text
v2.0.0
```

Recommended release title:

```text
Sediment Textural Analysis v2.0.0: EDUCOAST Educational Workflow
```

## 2. Test Before Release

```bash
pip install -r requirements.txt
python sediment_textural_graphical_moments_parameters.py
python sediment_textural_graphical_moments_plots.py
```

Confirm that:

- `output_textural_parameters_Data_base_EDUCOAST.xlsx` is created;
- figures are created in `figures/`;
- no personal local paths are present in the scripts;
- the README correctly describes the EDUCOAST dataset.

## 3. Create GitHub Release

Create a new release using tag `v2.0.0`.

## 4. Archive With Zenodo

After the GitHub release is created, Zenodo should archive the new version if the repository is linked.

Use the version-specific Zenodo DOI in the article.
