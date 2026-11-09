from __future__ import annotations

import argparse
import json

import numpy as np
from .data import GROUP_NAME, GROUP_SPLIT, make_dataset
from .model import evaluate, fit_models, subgroup_metrics
from .split import split_indices

PROJECT = "MedSignal-NoShow — Appointment No-Show Research Prototype"


def run(seed=42, samples=2400):
    X, y, groups = make_dataset(n=samples, seed=seed)
    split_groups = groups if GROUP_SPLIT else None
    tr, te = split_indices(y, seed=seed, groups=split_groups)
    models = fit_models(X[tr], y[tr], seed=seed)
    report = {
        "project": PROJECT,
        "data": "synthetic demonstration data",
        "seed": int(seed),
        "split": "group holdout" if GROUP_SPLIT else "stratified random holdout",
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "baseline": evaluate(models["baseline_prior"], X[te], y[te]),
        "stronger_model": evaluate(models["hist_gradient_boosting"], X[te], y[te]),
    }
    if GROUP_NAME and not GROUP_SPLIT:
        report["subgroup_diagnostic"] = {"attribute": GROUP_NAME,
            **subgroup_metrics(models["hist_gradient_boosting"], X[te], y[te], groups[te])}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate the synthetic research prototype.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=2400)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)
    report = run(args.seed, args.samples)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
