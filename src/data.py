from __future__ import annotations

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "hemoglobin",
    "mcv",
    "platelets",
    "systolic_bp",
    "glucose",
    "inflammation_marker",
]

CATEGORICAL_FEATURES = ["sex", "age_group", "risk_group"]


def make_synthetic_clinical_data(
    n_samples: int = 900,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create synthetic mixed-tabular data with injected anomaly labels.

    The feature names are clinical-style, but every value is generated from
    simple distributions. No real patient data or external files are used.
    """
    rng = np.random.default_rng(random_state)
    n_anomalies = max(1, int(round(n_samples * contamination)))
    n_normal = n_samples - n_anomalies

    normal = pd.DataFrame(
        {
            "hemoglobin": rng.normal(13.8, 1.1, n_normal),
            "mcv": rng.normal(88.0, 6.0, n_normal),
            "platelets": rng.normal(245.0, 45.0, n_normal),
            "systolic_bp": rng.normal(121.0, 12.0, n_normal),
            "glucose": rng.normal(96.0, 12.0, n_normal),
            "inflammation_marker": rng.gamma(shape=2.0, scale=1.8, size=n_normal),
            "sex": rng.choice(["female", "male"], size=n_normal, p=[0.52, 0.48]),
            "age_group": rng.choice(["young", "middle", "older"], size=n_normal, p=[0.25, 0.50, 0.25]),
            "risk_group": rng.choice(["low", "medium", "high"], size=n_normal, p=[0.62, 0.30, 0.08]),
        }
    )

    anomalies = pd.DataFrame(
        {
            "hemoglobin": rng.normal(9.3, 1.0, n_anomalies),
            "mcv": rng.normal(69.0, 5.0, n_anomalies),
            "platelets": rng.normal(390.0, 70.0, n_anomalies),
            "systolic_bp": rng.normal(158.0, 18.0, n_anomalies),
            "glucose": rng.normal(156.0, 26.0, n_anomalies),
            "inflammation_marker": rng.gamma(shape=5.0, scale=2.6, size=n_anomalies),
            "sex": rng.choice(["female", "male"], size=n_anomalies, p=[0.45, 0.55]),
            "age_group": rng.choice(["young", "middle", "older"], size=n_anomalies, p=[0.06, 0.24, 0.70]),
            "risk_group": rng.choice(["low", "medium", "high"], size=n_anomalies, p=[0.04, 0.20, 0.76]),
        }
    )

    data = pd.concat([normal, anomalies], ignore_index=True)
    labels = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_anomalies, dtype=int)])

    order = rng.permutation(n_samples)
    return data.iloc[order].reset_index(drop=True), labels[order]
