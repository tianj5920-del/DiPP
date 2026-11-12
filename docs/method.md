# Algorithm and reproduction scope

Probability column order is fixed as **HC, Prodromal, PD** (0,1,2), matching the experiment. Display tables may list HC, PD, Prodromal; this does not change array order.

## Protein features

Fit eligibility masks using the training subset only. Broad assays have at least80% finite NPX and at least5 observations per class. Strict quantitative assays additionally have at least80% paired NPX/LOD availability and at least50% detection. Detection channels require at least10 observations on each side of the LOD. The excess block uses `max(NPX-LOD,0)` for eligible detection assays not in the strict quantitative set. Quantitative channels use training medians; detection channels use training frequencies. All channels are standardized using training statistics, and constant quantitative channels are removed.

## Single-member E7 + HC kernel

1. Fit balanced multinomial LR with C=0.003, scaling expression/detection blocks by1/0.5.
2. Form two centered, energy-normalized LR contrast coordinates: HC−PD and Prodromal−PD. Add6/4/4 PCA coordinates from NPX/detection/excess, normalizing each PCA block and scaling their concatenation by1/sqrt(3).
3. Learn a16×16 linear residual update initialized at zero. Each step uses4 episodes, each with5 disjoint supports and5 queries per class. Optimize CE plus0.3 times six directed pairwise logistic ranking losses. Use800 AdamW steps at0.0003, weight decay0.001 and gradient clipping1. There is no teacher-KL term.
4. Estimate Gaussian means and population variances from supports in training and all training participants at inference. Shrink variance halfway toward the class-average variance; floor it at0.01 times mean pooled variance plus1e-6. Inference logits are LR logits plus1.5 times the learned-minus-reference Gaussian score.
5. Fit the HC kernel on unrescaled NPX/detection features. RBF bandwidth is mean off-diagonal squared training distance. Solve the class-balanced kernel system with the original192-step preconditioned conjugate-gradient algorithm; check relative residual≤1e-4. Convert scores with a sigmoid scale2.
6. The HC probability vector retains the prototype's conditional PD/Prodromal proportions. Fit two shared simplex weights on five-fold training OOF predictions, using balanced NLL and0.1 times mean squared raw-weight logits, Adam0.03 for300 steps. Refit the base branches on all training participants.

The seed for the full neural fit is101+7699. Inner fusion fits use model seeds101+200003×(fold+1), followed by the same7699 offset. Inner folds use49101, outer folds58111. Fold stratification uses class×project when feasible, otherwise class alone.

## Ablations and classifiers

`without_hc` retains the LR-guided prototype. `lr_hc` removes prototype correction but keeps the same LR and HC branches. `lr_free` and `lr_free_hc` never fit or use LR: they use8/4/4 PCA coordinates scaled bysqrt(2/3), a linear residual encoder and direct Gaussian scores. The two-branch variants each relearn fusion weights in training folds.

Classifiers use NPX/detection features, three-fold training OOF OvO AUC for hyperparameter selection, and a full-training refit. The paper grids and neural member seeds are defined in `baselines.py`. The portable pipeline optionally wraps classifier selection in the same outer validation loop used for the final model. Test metrics are computed after fitting and selection.

## Grouped SHAP

Each assay's complete set of input channels forms one Shapley player. A masked assay is jointly replaced by its channels from a training background participant. For each boundary, the background contains12 participants from each relevant class; the explained function is log(p_A)−log(p_B). The public SHAP `PermutationExplainer` uses paired forward/reverse permutations; paper configuration uses16 cycles. The implementation here uses the ordinary grouped masking interface rather than the experiment's coalition-cache optimization.

Rank by the equal-class mean absolute SHAP, then average equally across split-specific fitted models. Assays absent from one model contribute zero to that model's importance. Normalize each boundary's Top1 to100. Direction is the within-recipient-class mean attribution contrast between high NPX (≥training Q75) and low NPX (<training Q25), aggregated over available class/model cells. Missing direction support remains missing. Contribution magnitude is in log-score units, not an AUC change or causal concentration effect.

The release checks SHAP additivity for every explained synthetic assessment and reports cross-split rank agreement. The historical adaptive8→16-cycle and background12→24 sensitivity runs are not silently rerun by the default command; use separately configured runs to study these choices.

## Portability and historical reproduction

The normalized CSV adapter replaces private data readers, medication adjudication services, memory caches and experiment schedulers. The numerical model recipe is retained. Real training data, real participant allocations, fitted checkpoints and individual SHAP values are not distributed.

The automatic holdout generator makes **new class-stratified random splits**. Exact historical reproduction additionally requires the original authorized cohort, preprocessing inputs and frozen allocation files. Supply those via the explicit input contract and `--splits`; sharing random seed numbers alone is insufficient. Floating-point backend and PCA solver differences can also affect fitted scores. The synthetic test suite verifies numerical primitives and end-to-end execution; it is not a replay of the real-cohort performance.
