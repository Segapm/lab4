"""Обучение ансамбля и оценка на переданном тестовом наборе."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


CONTINUOUS_CREDIT_COLUMNS = ("Age_years", "Duration_of_Credit_monthly", "Credit_Amount")
LOG_MONTHLY_AMOUNT = "Log_Credit_Amount_per_month"
MODEL_NAME = "Soft Voting (Random Forest + MLP)"


def _feature_frame(features):
    if sparse.issparse(features):
        features = features.toarray()
    frame = pd.DataFrame(features).rename(columns=str)
    return frame.where(frame.notna(), np.nan)


def _numeric_columns(frame):
    if set(CONTINUOUS_CREDIT_COLUMNS).issubset(frame.columns):
        columns = list(CONTINUOUS_CREDIT_COLUMNS)
        if LOG_MONTHLY_AMOUNT in frame:
            columns.append(LOG_MONTHLY_AMOUNT)
        return columns

    numeric = []
    for column in frame.select_dtypes(include="number"):
        observed = frame[column].dropna().to_numpy(dtype=float)
        # Для таблиц без схемы небольшие наборы целых значений считаем кодами.
        if len(np.unique(observed)) > 12 or np.any(observed % 1 != 0):
            numeric.append(column)
    return numeric


def _categorical_columns(frame):
    numeric = set(_numeric_columns(frame))
    return [column for column in frame if column not in numeric]


def _preprocessor(*, one_hot):
    encoder = (
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if one_hot
        else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    )
    return ColumnTransformer([
        (
            "numeric",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            _numeric_columns,
        ),
        (
            "categorical",
            make_pipeline(SimpleImputer(strategy="most_frequent", keep_empty_features=True), encoder),
            _categorical_columns,
        ),
    ])


def _credit_features(frame, *, monthly_amount):
    if not set(CONTINUOUS_CREDIT_COLUMNS).issubset(frame.columns):
        return frame
    columns = list(CONTINUOUS_CREDIT_COLUMNS)
    values = frame[columns]
    result = frame.copy()
    result[columns] = np.log(values.where(values > 0))
    if monthly_amount:
        result[LOG_MONTHLY_AMOUNT] = (
            result["Credit_Amount"] - result["Duration_of_Credit_monthly"]
        )
    return result


def _forest(seed, *, min_samples_leaf=3, max_features=0.5, one_hot=False):
    return make_pipeline(
        _preprocessor(one_hot=one_hot),
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=1,
            random_state=seed,
        ),
    )


def _mlp(seed, *, features="monthly", layers=(12,), alpha=10.0):
    steps = []
    if features != "raw":
        steps.append(FunctionTransformer(
            _credit_features, kw_args={"monthly_amount": features == "monthly"}
        ))
    return make_pipeline(
        *steps,
        _preprocessor(one_hot=True),
        MLPClassifier(
            hidden_layer_sizes=layers,
            activation="tanh",
            solver="lbfgs",
            alpha=alpha,
            max_iter=1000,
            random_state=seed,
        ),
    )


def _selected_model(seed):
    return VotingClassifier(
        estimators=[("forest", _forest(seed)), ("mlp", _mlp(seed))],
        voting="soft",
        n_jobs=1,
    )


def _metrics(target, probabilities, predictions):
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучает ансамбль на train и возвращает оценки положительного класса 1."""
    train = _feature_frame(data.train_features)
    test = _feature_frame(data.test_features)
    if not isinstance(data.train_features, pd.DataFrame) or not isinstance(
        data.test_features, pd.DataFrame
    ):
        test.columns = train.columns

    model = _selected_model(data.random_seed)
    model.fit(train, np.asarray(data.train_target).ravel())
    positive_column = np.flatnonzero(model.classes_ == 1).item()
    probabilities = model.predict_proba(test)[:, positive_column]
    predictions = (probabilities >= 0.5).astype(int)
    return EnsembleResult(
        model_name=MODEL_NAME,
        probabilities=probabilities,
        predictions=predictions,
        **_metrics(np.asarray(data.test_target).ravel(), probabilities, predictions),
    )
