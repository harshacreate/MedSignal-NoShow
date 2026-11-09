# MedSignal-NoShow — Appointment No-Show Research Prototype

**Sector:** Healthcare  
**Task:** Binary classification: scheduled appointment no-show

A synthetic appointment-level classification exercise using lead time, prior no-shows, reminders, chronic-condition follow-up, distance, age, weekday, and first-visit status.

## Data and provenance

The offline demo uses a deterministic synthetic generator in `medsignal/data.py`. No public or private dataset is downloaded or redistributed. The generated values are illustrative and do not represent real people, organizations, or operational systems. Synthetic output is not evidence of real-world model performance.

## Method

- **Baseline:** majority-class prior classifier.
- **Stronger model:** histogram gradient-boosting classifier.
- **Split:** Stratified random holdout. Each synthetic row is independently generated, so a random split matches the generator design; this is not a temporal deployment evaluation.. The seed is fixed by default and configurable.
- **Evaluation:** accuracy, balanced accuracy, macro F1, confusion matrix, and (for binary tasks) ROC-AUC, average precision, precision, and recall at threshold 0.5.
- **Error analysis:** confusion matrix and misclassification count are emitted with the metrics.

Research/education only. No patient records are used. A reminder or care decision must remain under qualified human review. Age-band recall and prediction-rate gaps are descriptive diagnostics, not proof of fairness.

## Run locally

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -e ".[test]"
    python -m medsignal.train --seed 42 --samples 2400
    python -m medsignal.train --seed 42 --samples 2400 --json
    python -m pytest -q

`make setup`, `make train`, and `make test` provide shortcuts. Python 3.9 or newer is required. No network access or data download is needed after dependencies are installed.

## Repository layout

- `medsignal/data.py` — validated domain-specific synthetic generator.
- `medsignal/split.py` — repeatable holdout split.
- `medsignal/model.py` — baseline, stronger model, metrics, and diagnostic summaries.
- `medsignal/train.py` — train/evaluate CLI.
- `tests/` — validation, split, model, and CLI tests.
- `.github/workflows/ci.yml` — automated test and CLI smoke run.

## Intended use

Educational and research prototype only. Do not use the synthetic benchmark to make decisions about real people or operations. Review the data assumptions, error patterns, and subgroup diagnostics before describing any findings.
