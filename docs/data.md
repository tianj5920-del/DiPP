# Input tables

Place these three UTF-8 CSV files in one directory. Identifiers are strings. Missing numeric values are empty cells. The example's `SYN_*` identifiers and `SyntheticProtein*` names are entirely fictitious.

## participants.csv

| Column | Meaning |
| --- | --- |
| PATNO | Participant key, unique in this table |
| ENROLL_CAT | HC, Prodromal or PD; NC is accepted as an alias of HC |
| SEX | Participant metadata, e.g. M/F; never a predictor |
| DM_CORE_EVIDENCE | 0/1: adjudicated core diabetes medication/indication evidence over the study observation window |
| GLUCOSE_MMOL_L | Valid glucose value/max in mmol/L for the eligibility rule; missing is permitted |

Eligibility is `DM_CORE_EVIDENCE == 1 OR GLUCOSE_MMOL_L >= 7.0`, in an allowed enrollment class. This follows the research cohort's fasting-based eligibility assumption; the input flag is not a newly inferred diabetes diagnosis. Clinical variables determine membership and are not concatenated with protein predictors.

## assays.csv

| Column | Meaning |
| --- | --- |
| OlinkID | Unique assay identifier; do not merge same-named assays |
| Assay | Display name of the measured protein |

## plasma.csv

One row per participant × project × assay:

| Column | Meaning |
| --- | --- |
| PATNO, OlinkID | Keys matching the metadata tables |
| PROJECT | 314 or 293 |
| EVENT_ID | BL for eligible baseline samples |
| BIOFLUID | Plasma |
| SampleType | SAMPLE for biological participant samples |
| AssayType | assay for protein measurements |
| NPX | Released relative protein expression on the log2 scale |
| LOD | Corresponding assay/sample limit of detection in NPX units |
| QC_Pass | 0/1: combined, already-reviewed sample and assay QC decision |

Convert official exports to this contract by mapping their metadata, visit/biofluid fields, NPX/LOD columns, and technical QC decisions. Resolve technical replicates explicitly before import; duplicate participant/project/assay rows are rejected. The adapter intentionally does not assume that every PPMI export version uses the same clinical or QC codes.

Processing order:

1. Keep biological baseline plasma from projects314/293.
2. Replace failed-QC values with missing values.
3. Establish the common measured assay catalog.
4. Require at least80% valid NPX coverage for each candidate participant/project sample.
5. Apply clinical eligibility, prefer a passing314 sample, otherwise use a passing293 sample.
6. Fit all assay selection, imputation, scaling, PCA and prediction parameters within training folds.

No cross-project bridge adjustment is applied in the final recipe. In `features.py`, missing detection values remain unknown until imputed using training detection frequency; they are not assigned to the below-LOD category.

## Optional frozen allocations

`--splits allocations.csv` accepts `PATNO,seed,split`, where split is train/test. Every eligible participant must occur once for every requested seed, and no extra participant keys are permitted. Keep real allocations and real data in your private workspace; this release contains only synthetic examples.
