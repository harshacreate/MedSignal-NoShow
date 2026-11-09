from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def split_indices(y, seed=42, groups=None, test_size=0.25):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) < 8:
        raise ValueError("y must be a one-dimensional array with at least 8 rows")
    indices = np.arange(len(y))
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != y.shape or len(np.unique(groups)) < 2:
            raise ValueError("groups must align with y and contain at least two groups")
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(indices, y, groups))
        return train, test
    stratify = y if len(np.unique(y)) > 1 and np.min(np.unique(y, return_counts=True)[1]) >= 2 else None
    return train_test_split(indices, test_size=test_size, random_state=seed, stratify=stratify)
