"""Deterministic, domain-specific synthetic dataset for MedSignal-NoShow — Appointment No-Show Research Prototype."""

from __future__ import annotations
import numpy as np

FEATURES = ['age', 'lead', 'prior', 'sms', 'chronic', 'distance', 'dow', 'first']
GROUP_NAME = 'age_band'
GROUP_SPLIT = False

def make_dataset(n=2400, seed=42):
    if n < 80:
        raise ValueError("n must be at least 80 for a useful holdout")
    rng = np.random.default_rng(seed)
    age = rng.integers(1, 95, n).astype(float)
    lead = rng.integers(0, 90, n).astype(float)
    prior = rng.poisson(0.6, n).astype(float)
    sms = rng.integers(0, 2, n).astype(float)
    chronic = rng.integers(0, 2, n).astype(float)
    distance = np.clip(rng.exponential(8, n), 0, 60)
    dow = rng.integers(0, 7, n).astype(float)
    first = rng.integers(0, 2, n).astype(float)
    z = -1.7 + .03*lead + .75*prior - 1.15*sms - .55*chronic + .024*distance + .45*first - .013*np.maximum(age-40, 0) + .18*(dow >= 5)
    p = 1/(1+np.exp(-z))
    y = (rng.random(n) < p).astype(int)
    groups = np.where(age < 50, "under_50", "50_plus")
    X = np.column_stack([age, lead, prior, sms, chronic, distance, dow, first]).astype(float)
    if X.shape != (n, len(FEATURES)) or not np.isfinite(X).all():
        raise ValueError("generator produced invalid feature matrix")
    if np.asarray(groups).shape != (n,):
        raise ValueError("generator produced invalid group vector")
    return X, y.astype(int), np.asarray(groups)
