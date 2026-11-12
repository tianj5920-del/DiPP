# Synthetic example

All participants, assay names, identifiers and values in this directory are invented.
The files illustrate the package's normalized PPMI-style input contract, not an official PPMI export.

- 216 eligible participants:48 HC,96 Prodromal,72 PD.
- 64 synthetic assays, with known toy class shifts, correlations, missing values and QC flags.
- Eight participants appear in both projects to exercise314/293 sample priority.
- Three extra participants exercise exclusion for a nonbaseline visit, failed sample coverage, and absent diabetes-related eligibility.

`SYNTHETIC_DATA.json` records the generator seed and file hashes. No real records or real-data distribution estimates were used. Run `dipp-label make-example --out dataset/example --overwrite` to reproduce these files.
