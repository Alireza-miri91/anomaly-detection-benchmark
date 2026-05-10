from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
from adadmire import admire, penalty
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM

from data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def robust_zscore_scores(data: pd.DataFrame) -> np.ndarray:
    numeric = data[NUMERIC_FEATURES].to_numpy()
    median = np.median(numeric, axis=0)
    mad = np.median(np.abs(numeric - median), axis=0)
    mad = np.where(mad == 0, 1.0, mad)
    zscores = 0.6745 * np.abs(numeric - median) / mad
    return zscores.max(axis=1)


def adadmire_scores(data: pd.DataFrame) -> np.ndarray:
    """Run public adADMIRE and convert flagged cell positions into row scores."""
    numeric = StandardScaler().fit_transform(data[NUMERIC_FEATURES])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    discrete = encoder.fit_transform(data[CATEGORICAL_FEATURES])
    levels = np.array([len(categories) for categories in encoder.categories_])

    lambda_sequence = penalty(numeric, discrete, min=-2.25, max=-1.5, step=0.25)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _, n_cont, position_cont, _, n_disc, position_disc = admire(
            numeric,
            discrete,
            levels,
            lambda_sequence,
            oIterations=1000,
        )

    scores = np.zeros(data.shape[0], dtype=float)
    if n_cont:
        continuous_positions = np.asarray(position_cont)
        continuous_rows = continuous_positions[:, 0].astype(int)
        np.add.at(scores, continuous_rows, 1.0)
    if n_disc:
        discrete_positions = np.asarray(position_disc)
        discrete_rows = discrete_positions[0, :].astype(int)
        np.add.at(scores, discrete_rows, 1.0)

    return scores


def run_anomaly_methods(
    data: pd.DataFrame,
    contamination: float,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Fit anomaly detectors and return scores where higher means more anomalous."""
    methods: dict[str, np.ndarray] = {}

    methods["adADMIRE"] = adadmire_scores(data)

    isolation_forest = Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            (
                "model",
                IsolationForest(
                    n_estimators=200,
                    contamination=contamination,
                    random_state=random_state,
                ),
            ),
        ]
    )
    isolation_forest.fit(data)
    methods["Isolation Forest"] = -isolation_forest.named_steps["model"].score_samples(
        isolation_forest.named_steps["preprocess"].transform(data)
    )

    one_class_svm = Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            (
                "model",
                OneClassSVM(kernel="rbf", nu=contamination, gamma="scale"),
            ),
        ]
    )
    one_class_svm.fit(data)
    methods["One-Class SVM"] = -one_class_svm.named_steps["model"].score_samples(
        one_class_svm.named_steps["preprocess"].transform(data)
    )

    lof = Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            (
                "model",
                LocalOutlierFactor(
                    n_neighbors=35,
                    contamination=contamination,
                    novelty=True,
                ),
            ),
        ]
    )
    lof.fit(data)
    methods["Local Outlier Factor"] = -lof.named_steps["model"].score_samples(
        lof.named_steps["preprocess"].transform(data)
    )

    methods["Robust Z-Score"] = robust_zscore_scores(data)
    return methods
