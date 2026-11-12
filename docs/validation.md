# Release validation

Date: 2026-11-12. Release: 0.1.0. Backend: **CPU**.

## Executed checks

- **24 tests passed**, including agreement with frozen LR, Gaussian-score, episode-ranking, sampler and kernel-solver functions.
- Verified train/test participant disjointness, train-only preprocessing, query-label independence, fully LR-free branches, fixed probability-column order and HC conditional-ratio preservation.
- Matched a two-group SHAP example against exact Shapley values; checked additivity for the complete fitted model and zero contribution for assays absent from a split-specific model.
- Exercised LightGBM and XGBoost in isolated CPU workers after detecting a shared-runtime conflict in an initial integration attempt.
- Ran the complete synthetic pipeline for **15 model/reference configurations**, producing **30 CV/test metric records**. This includes the final model, four ablations, nine standard classifiers and the analytical random reference.
- The full demonstration used 216 eligible synthetic participants and 64 synthetic assays, three outer validation folds, three inner fusion folds, and the quick-start optimization budgets. Wall time was approximately **345 seconds** in the tested environment.
- SHAP explained six synthetic recipients per boundary, with four training background samples per class and two permutation cycles. Maximum additive reconstruction error across the three boundaries was **6.32e-07** in log-score units.

The machine-readable settings and installed versions are in [validation.json](validation.json).
The published synthetic generator also covers participant exclusions, project priority, missing values and QC masking.

## Scope

The checks cover the portable CPU package and invented data. The full 800-step paper recipe and the real-cohort numerical results were not replayed in this release preparation. Private experiment schedulers, raw-data readers, participant allocations, model checkpoints and individual predictions are not shipped.

Run the checks yourself:

```bash
python -m pytest -q
dipp-label run --data dataset/example --config configs/example.json \
  --out outputs/release_check --models dipp_label without_hc lr_hc lr_free_hc lr_free \
  lr linear_svm rbf_svm random_forest extra_trees lightgbm xgboost mlp residual_mlp random --shap
```
