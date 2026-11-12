from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from . import CLASSES


def folds(cohort, count, seed):
    if min(np.bincount(cohort.y, minlength=3)) < count:
        raise ValueError("Too few participants per class for requested folds")
    source = np.unique(cohort.project, return_inverse=True)[1]
    strata = 3 * source + cohort.y
    if min(Counter(strata.tolist()).values()) < count:
        strata = cohort.y
    return list(StratifiedKFold(count, shuffle=True, random_state=seed).split(np.zeros(len(cohort)), strata))


def holdout(cohort, seed, fraction=.2, test_counts=None):
    """A NEW reproducible class-stratified random split, not the historical identities."""
    rng = np.random.default_rng(seed)
    selected = []
    for label, name in enumerate(CLASSES):
        rows = np.flatnonzero(cohort.y == label)
        count = (test_counts or {}).get(name, max(1, int(round(len(rows) * fraction))))
        if count < 1 or count >= len(rows):
            raise ValueError(f"Invalid test allocation for {name}")
        selected.extend(rng.choice(rows, count, replace=False))
    test = np.sort(selected)
    train = np.setdiff1d(np.arange(len(cohort)), test)
    return train, test


def read_holdout(cohort, path, seed):
    frame = pd.read_csv(path, dtype={"PATNO": str})
    if not {"PATNO", "seed", "split"}.issubset(frame):
        raise ValueError("Split CSV needs PATNO, seed, split")
    frame = frame[frame.seed == seed]
    if frame.PATNO.duplicated().any() or set(frame.PATNO) != set(cohort.ids):
        raise ValueError("Each eligible participant must appear exactly once in each supplied split")
    if set(frame.split) != {"train", "test"}:
        raise ValueError("Split labels must be train/test")
    by_id = frame.set_index("PATNO").split
    assignment = by_id.loc[cohort.ids].to_numpy()
    return np.flatnonzero(assignment == "train"), np.flatnonzero(assignment == "test")
