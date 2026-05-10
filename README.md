# Anomaly Detection Benchmark

This is a small, public-safe anomaly-detection demo for mixed tabular data. It mirrors the structure of a clinical-style benchmark from my CV, but it uses only synthetic data generated at runtime. No patient data, private course files, unpublished notes, or credentials are included.

## What This Project Shows

- Synthetic mixed-tabular data generation with numeric and categorical features.
- Injected anomalies with known labels for fair evaluation.
- Comparison of public `adADMIRE`, Isolation Forest, One-Class SVM, Local Outlier Factor, and a robust z-score baseline.
- Imbalanced evaluation with PR-AUC, ROC-AUC, and Precision@k.
- Reproducible outputs saved as JSON and plots.

## Repository Structure

```text
anomaly-detection-benchmark/
  README.md
  requirements.txt
  .gitignore
  src/
    data.py
    models.py
    evaluate.py
    plots.py
    main.py
  outputs/
    .gitkeep
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

The default run includes `adADMIRE`, which performs leave-one-out model fitting and may take around one to two minutes on a laptop.

Optional arguments:

```bash
python src/main.py --n-samples 1200 --contamination 0.03 --random-state 7
```

## Outputs

The script prints a compact metric table and writes:

- `outputs/metrics.json`
- `outputs/precision_recall.png`
- `outputs/score_distribution.png`

`Precision@k` uses `k = number of injected anomalies` by default, which is a practical way to ask: "If we reviewed the top flagged rows, how many would be true anomalies?"

## Public-Safe Scope

The feature names are clinical-style, but all values are generated from simple random distributions. This repository uses the public MIT-licensed `adadmire` package from Spang Lab and intentionally does not include private lab files, real clinical records, course datasets, notebooks, or generated research artifacts.
