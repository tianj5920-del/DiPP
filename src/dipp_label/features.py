"""Training-only NPX, detection and low-detection excess encodings."""
import numpy as np
from sklearn.preprocessing import StandardScaler


class ProteinFeatures:
    def __init__(self, config):
        self.config = config

    def fit(self, cohort):
        x, lod, y, c = cohort.npx, cohort.lod, cohort.y, self.config
        if set(y) != {0, 1, 2}:
            raise ValueError("Feature fitting requires all three training classes")
        valid = np.isfinite(x)
        paired = valid & np.isfinite(lod)
        broad = valid.mean(0) >= c.assay_coverage
        for label in range(3):
            broad &= valid[y == label].sum(0) >= c.min_per_class
        detected = (paired & (x >= lod)).sum(0)
        below = (paired & (x < lod)).sum(0)
        strict = broad & (paired.mean(0) >= c.lod_coverage) & (detected / np.maximum(paired.sum(0), 1) >= c.detection_min)
        binary = broad & (paired.mean(0) >= c.lod_coverage) & (detected >= c.binary_minimum) & (below >= c.binary_minimum)
        candidates = [np.flatnonzero(strict), np.flatnonzero(binary), np.flatnonzero(binary & ~strict)]
        self.blocks, self.fill_values, self.scalers = [], [], []
        self.assay_ids = cohort.assay_ids.copy()
        for channel, columns in enumerate(candidates):
            values = self._values(cohort, columns, channel)
            if values.shape[1]:
                fill = np.nanmean(values, axis=0) if channel == 1 else np.nanmedian(values, axis=0)
                filled = np.where(np.isfinite(values), values, fill)
                keep = np.ones(len(columns), bool) if channel == 1 else filled.std(0) > 1e-8
                columns, fill, filled = columns[keep], fill[keep], filled[:, keep]
                scaler = StandardScaler().fit(filled) if len(columns) else None
            else:
                fill, scaler = np.array([]), None
            self.blocks.append(columns)
            self.fill_values.append(fill)
            self.scalers.append(scaler)
        if not len(self.blocks[0]):
            raise ValueError("No variable, well-detected quantitative assays in training fold")
        self.widths = [len(b) for b in self.blocks]
        self.column_assays = np.concatenate(self.blocks)
        self.column_channels = np.concatenate([np.repeat(s, len(b)) for s, b in zip(("NPX", "detection", "excess"), self.blocks)])
        return self

    @staticmethod
    def _values(cohort, columns, channel):
        x, lod = cohort.npx[:, columns], cohort.lod[:, columns]
        if channel == 0:
            return x
        paired = np.isfinite(x) & np.isfinite(lod)
        return np.where(paired, (x >= lod).astype(float) if channel == 1 else np.maximum(x - lod, 0), np.nan)

    def transform(self, cohort):
        if not np.array_equal(cohort.assay_ids, self.assay_ids):
            raise ValueError("Assay catalog/order does not match fitted preprocessing")
        parts = []
        for channel, (columns, fill, scaler) in enumerate(zip(self.blocks, self.fill_values, self.scalers)):
            values = self._values(cohort, columns, channel)
            parts.append(scaler.transform(np.where(np.isfinite(values), values, fill)) if scaler is not None else values)
        out = np.concatenate(parts, axis=1).astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite encoded feature values")
        return out

    def fit_transform(self, cohort):
        return self.fit(cohort).transform(cohort)
