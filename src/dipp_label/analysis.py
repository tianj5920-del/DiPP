"""Grouped SHAP for frozen model scores, with training-only backgrounds."""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from . import CLASSES, PAIRS


def pair_scores(p, pair):
    p = np.maximum(p, 1e-12)
    return np.log(p[:, pair[0]]) - np.log(p[:, pair[1]])


class AssayMasker:
    """SHAP's boolean-mask interface, with one player for all columns of an assay."""
    def __init__(self, background, query, groups):
        self.background = background
        self.query = query
        self.groups = groups
        self.shape = (len(background), len(groups))
        self.clustering = None

    def __call__(self, mask, x):
        masked = self.background.copy()
        for selected, columns in zip(mask, self.groups):
            if selected:
                masked[:, columns] = self.query[columns]
        return (masked,)

    def invariants(self, x):
        return np.stack([np.all(self.background[:, columns] == self.query[columns], axis=1) for columns in self.groups], axis=1)


def grouped_shap(model, training, testing, config):
    import shap
    encoder = model.base.features
    train = encoder.transform(training)
    test = encoder.transform(testing)
    assays = np.unique(encoder.column_assays)
    groups = [np.flatnonzero(encoder.column_assays == a) for a in assays]
    output, diagnostics = [], []
    for pair_index, (a, b) in enumerate(PAIRS):
        name = f"{CLASSES[a]}_vs_{CLASSES[b]}"
        rng = np.random.default_rng(config.shap_seed + pair_index)
        backgrounds = []
        for label in (a, b):
            pool = np.flatnonzero(training.y == label)
            if len(pool) < config.shap_background_per_class:
                raise ValueError("Insufficient training participants for SHAP background")
            backgrounds.extend(rng.choice(pool, config.shap_background_per_class, replace=False))
        recipients = []
        for label in (a, b):
            pool = np.flatnonzero(testing.y == label)
            if config.shap_max_per_class is not None:
                pool = rng.choice(pool, min(len(pool), config.shap_max_per_class), replace=False)
            recipients.extend(pool)
        recipients = np.array(recipients, dtype=int)
        values, errors = [], []
        def function(encoded):
            return pair_scores(model.predict_encoded(encoded), (a, b))
        for index in recipients:
            masker = AssayMasker(train[backgrounds], test[index], groups)
            explainer = shap.PermutationExplainer(function, masker,
                feature_names=training.assay_ids[assays].tolist(), seed=config.shap_seed + pair_index * 100003 + int(index))
            result = explainer(np.ones((1, len(groups))), max_evals=(2 * len(groups) + 1) * config.shap_cycles, silent=True)
            row = np.asarray(result.values[0], dtype=float)
            error = abs(float(result.base_values[0] + row.sum()) - float(function(test[index:index + 1])[0]))
            if error > 1e-4:
                raise ValueError("SHAP additive reconstruction failed")
            errors.append(error)
            values.append(row)
        values = np.asarray(values)
        labels = testing.y[recipients]
        # Each recipient class contributes one half regardless of class size.
        importance = .5 * (np.abs(values[labels == a]).mean(0) + np.abs(values[labels == b]).mean(0))
        for j, assay in enumerate(assays):
            contrasts = []
            train_npx = training.npx[:, assay]
            low, high = np.nanquantile(train_npx, [.25, .75])
            for label in (a, b):
                observed = testing.npx[recipients, assay]
                upper = (labels == label) & np.isfinite(observed) & (observed >= high)
                lower = (labels == label) & np.isfinite(observed) & (observed < low)
                if upper.any() and lower.any():
                    contrasts.append(float(values[upper, j].mean() - values[lower, j].mean()))
            output.append(dict(boundary=name, assay_id=training.assay_ids[assay], protein=training.assay_names[assay],
                mean_abs_shap=float(importance[j]), direction_contrasts=contrasts,
                channels="+".join(encoder.column_channels[groups[j]])))
        diagnostics.append(dict(boundary=name, recipient_count=len(recipients), background_per_class=config.shap_background_per_class,
                                cycles=config.shap_cycles, maximum_additivity_error=max(errors), assay_count=len(assays)))
    return output, diagnostics


def summarize_rankings(by_seed):
    rows, stability = [], []
    for a, b in PAIRS:
        boundary = f"{CLASSES[a]}_vs_{CLASSES[b]}"
        maps = [{r['assay_id']: r for r in records if r['boundary'] == boundary} for records in by_seed]
        ids = sorted(set.union(*(set(m) for m in maps)))
        for assay in ids:
            present = [m[assay] for m in maps if assay in m]
            contributions = [m.get(assay, {}).get('mean_abs_shap', 0.) for m in maps]
            contrasts = [v for r in present for v in r['direction_contrasts']]
            direction = None if not contrasts or np.mean(contrasts) == 0 else CLASSES[a if np.mean(contrasts) > 0 else b]
            agreement = None if not contrasts else max(sum(v > 0 for v in contrasts), sum(v < 0 for v in contrasts)) / len(contrasts)
            rows.append(dict(boundary=boundary, assay_id=assay, protein=present[0]['protein'],
                mean_abs_shap=float(np.mean(contributions)), sd_abs_shap=float(np.std(contributions, ddof=1)) if len(maps) > 1 else 0.,
                present_models=len(present), higher_npx_direction=direction, direction_agreement=agreement,
                direction_cells=len(contrasts), channels=present[0]['channels']))
        for i, j in itertools.combinations(range(len(maps)), 2):
            common = sorted(set(maps[i]) & set(maps[j]))
            left = [maps[i][k]['mean_abs_shap'] for k in common]
            right = [maps[j][k]['mean_abs_shap'] for k in common]
            rank = float(spearmanr(left, right).statistic) if len(common) > 1 and np.std(left) > 0 and np.std(right) > 0 else None
            tops = [set(sorted(m, key=lambda k:m[k]['mean_abs_shap'], reverse=True)[:10]) for m in (maps[i], maps[j])]
            stability.append(dict(boundary=boundary, model_pair=[i, j], common_assays=len(common), spearman=rank,
                                  top10_overlap=len(tops[0] & tops[1])/max(1, min(map(len,tops)))))
    frame = pd.DataFrame(rows).sort_values(['boundary', 'mean_abs_shap', 'assay_id'], ascending=[True, False, True])
    frame['rank'] = frame.groupby('boundary').cumcount() + 1
    maximum = frame.groupby('boundary').mean_abs_shap.transform('max')
    frame['normalized_percent'] = np.where(maximum > 0, frame.mean_abs_shap / maximum * 100, 0.)
    return frame, stability
