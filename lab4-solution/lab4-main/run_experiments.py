"""Повторяет сравнение на train и записывает результат одного запуска на test."""

import json
import os
import platform
from pathlib import Path

# Ограничиваем число потоков библиотек линейной алгебры до их импорта.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline

from ensemble_models import _forest, _metrics, _mlp, _preprocessor, run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput


ROOT = Path(__file__).resolve().parent
SEED = 42
TEST_SIZE = 0.2
FOLDS = 5
REPEATS = 2
SELECTED = "vote_rf_half_mlp_rf50"
METRIC_NAMES = ("roc_auc", "accuracy", "precision", "recall", "f1")


def candidates():
    return {
        "rf_sqrt_leaf3": _forest(SEED, max_features="sqrt"),
        "rf_sqrt_leaf1": _forest(SEED, min_samples_leaf=1, max_features="sqrt"),
        "rf_sqrt_leaf10": _forest(SEED, min_samples_leaf=10, max_features="sqrt"),
        "rf_half_leaf3": _forest(SEED),
        "rf_half_leaf5": _forest(SEED, min_samples_leaf=5),
        "rf_onehot_sqrt_leaf3": _forest(SEED, max_features="sqrt", one_hot=True),
        "extra_trees_leaf10": make_pipeline(
            _preprocessor(one_hot=False),
            ExtraTreesClassifier(
                n_estimators=300, min_samples_leaf=10, max_features=0.7,
                n_jobs=1, random_state=SEED,
            ),
        ),
        **{
            f"gb_depth{depth}_{trees}": make_pipeline(
                _preprocessor(one_hot=True),
                GradientBoostingClassifier(
                    n_estimators=trees, max_depth=depth, min_samples_leaf=12,
                    learning_rate=0.05, random_state=SEED,
                ),
            )
            for depth, trees in ((2, 100), (2, 250), (3, 250))
        },
        "mlp_raw_l2_10": _mlp(SEED, features="raw"),
        "mlp_log_l2_10": _mlp(SEED, features="log"),
        "mlp_monthly_l2_3": _mlp(SEED, alpha=3),
        "mlp_monthly_l2_10": _mlp(SEED),
        "mlp_monthly_l2_20": _mlp(SEED, alpha=20),
        "mlp_monthly_deeper_l2_10": _mlp(SEED, layers=(24, 8)),
    }


def summarize_folds(fold_targets, fold_probabilities):
    scores = [
        _metrics(target, probability, (probability >= 0.5).astype(int))
        for target, probability in zip(fold_targets, fold_probabilities)
    ]
    return {
        **{name: float(np.mean([row[name] for row in scores])) for name in METRIC_NAMES},
        "roc_auc_std": float(np.std([row["roc_auc"] for row in scores])),
    }


def cross_validation(train, target):
    cv = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED)
    folds = list(cv.split(train, target))
    fold_targets = [target.iloc[validation] for _, validation in folds]
    probabilities = {}
    for name, model in candidates().items():
        probabilities[name] = np.array([
            clone(model).fit(train.iloc[fit], target.iloc[fit]).predict_proba(
                train.iloc[validation]
            )[:, 1]
            for fit, validation in folds
        ])

    mlp = probabilities["mlp_monthly_l2_10"]
    for name, tree in (
        ("rf_half", "rf_half_leaf3"),
        ("rf_sqrt", "rf_sqrt_leaf3"),
        ("rf_full", "rf_sqrt_leaf1"),
    ):
        for weight in (25, 50, 75):
            portion = weight / 100
            probabilities[f"vote_{name}_mlp_rf{weight}"] = (
                portion * probabilities[tree] + (1 - portion) * mlp
            )
    probabilities["vote_gb_mlp_gb25"] = 0.25 * probabilities["gb_depth2_250"] + 0.75 * mlp

    return {
        name: summarize_folds(fold_targets, values) for name, values in probabilities.items()
    }


def main():
    features = pd.read_csv(ROOT / "german.csv", sep=";")
    target = features.pop("Creditability")
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=SEED, stratify=target
    )
    cv_metrics = cross_validation(x_train, y_train)
    winner = max(cv_metrics, key=lambda name: cv_metrics[name]["roc_auc"])

    result = run_ensemble_experiment(EnsembleInput(x_train, y_train, x_test, y_test, SEED))
    results = {
        "model_name": result.model_name,
        **{name: getattr(result, name) for name in METRIC_NAMES},
        "split": {
            "random_seed": SEED,
            "test_size": TEST_SIZE,
            "stratified": True,
            "train_rows": len(y_train),
            "test_rows": len(y_test),
        },
        "positive_label": 1,
        "decision_threshold": 0.5,
        "selection": {
            "criterion": "mean validation ROC-AUC",
            "folds": FOLDS,
            "repeats": REPEATS,
            "random_seed": SEED,
            "selected_experiment": SELECTED,
            "best_validation_experiment": winner,
        },
        "validation_experiments": cv_metrics,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    output = ROOT / "experiment_results.json"
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{result.model_name} | CV ROC-AUC: {cv_metrics[SELECTED]['roc_auc']:.4f}")
    if winner != SELECTED:
        print(f"Максимум ROC-AUC в текущем запуске у {winner}: {cv_metrics[winner]['roc_auc']:.4f}")
    print("Test: " + " | ".join(f"{name}: {results[name]:.4f}" for name in METRIC_NAMES))
    print(f"Метрики записаны в {output.name}")


if __name__ == "__main__":
    main()
