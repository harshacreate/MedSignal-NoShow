import numpy as np
from medsignal.data import FEATURES, make_dataset
from medsignal.train import run

def test_dataset_shape_and_balance():
    X, y, groups = make_dataset(n=3000, seed=1)
    assert X.shape == (3000, len(FEATURES))
    assert set(np.unique(y)) <= {0, 1}
    assert 0.1 < y.mean() < 0.6
    assert set(np.unique(groups)) == {"under_50", "50_plus"}

def test_model_has_signal_and_subgroup_report():
    report = run(seed=0)
    assert report["stronger_model"]["roc_auc"] >= 0.72, report
    assert report["stronger_model"]["recall_at_0_5"] >= 0.2
    assert report["subgroup_diagnostic"]["attribute"] == "age_band"
