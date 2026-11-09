import json
import os
import subprocess
import sys

import numpy as np

from medsignal.data import FEATURES, GROUP_NAME, GROUP_SPLIT, make_dataset
from medsignal.split import split_indices
from medsignal.train import run


def test_generator_is_reproducible_and_valid():
    X, y, groups = make_dataset(n=300, seed=17)
    X2, y2, groups2 = make_dataset(n=300, seed=17)
    assert X.shape == (300, len(FEATURES))
    assert np.isfinite(X).all()
    assert np.array_equal(y, y2)
    assert np.array_equal(X, X2)
    assert groups.shape == (300,)
    assert set(np.unique(y)).issubset(set(range(len(np.unique(y)))))


def test_split_respects_declared_strategy():
    X, y, groups = make_dataset(n=600, seed=11)
    tr, te = split_indices(y, seed=5, groups=groups if GROUP_SPLIT else None)
    assert not set(tr).intersection(te)
    if GROUP_SPLIT:
        assert not set(groups[tr]).intersection(groups[te])
    else:
        assert abs(float(np.mean(y[tr])) - float(np.mean(y[te]))) < 0.08


def test_models_return_valid_metrics():
    report = run(seed=9, samples=480)
    for key in ("baseline", "stronger_model"):
        assert report[key]["n"] > 0
        assert 0 <= report[key]["accuracy"] <= 1
        assert 0 <= report[key]["macro_f1"] <= 1
        assert report[key]["error_count"] >= 0
    if "subgroup_diagnostic" in report:
        assert report["subgroup_diagnostic"]["attribute"] == GROUP_NAME


def test_cli_json_smoke():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.getcwd(), env.get("PYTHONPATH", "")])
    result = subprocess.run([sys.executable, "-m", "medsignal.train", "--samples", "320", "--seed", "3", "--json"],
                            check=True, capture_output=True, text=True, env=env)
    payload = json.loads(result.stdout)
    assert payload["project"] == "MedSignal-NoShow — Appointment No-Show Research Prototype"
    assert payload["n_test"] > 0
