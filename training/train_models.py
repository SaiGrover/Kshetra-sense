"""Train and package Kshetra Sense's leakage-free crop classifier.

Model selection happens exclusively inside the training split. The held-out
test partition is evaluated once, after the winning estimator is calibrated.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
TEST_SIZE = 0.20
CONFIDENCE_THRESHOLD = 0.55
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "Crop_recommendation.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_eda(df: pd.DataFrame) -> dict:
    statistics, histograms, boxplots = {}, {}, {}
    for col in FEATURES:
        series = df[col]
        statistics[col] = {
            "mean": round(float(series.mean()), 4), "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4), "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4), "q01": round(float(series.quantile(0.01)), 4),
            "q25": round(float(series.quantile(0.25)), 4), "q75": round(float(series.quantile(0.75)), 4),
            "q99": round(float(series.quantile(0.99)), 4),
        }
        counts, edges = np.histogram(series, bins=20)
        histograms[col] = {"counts": counts.tolist(), "bin_edges": edges.round(4).tolist()}
        boxplots[col] = {key: statistics[col][key] for key in ("min", "q25", "median", "q75", "max")}
    return {
        "shape": list(df.shape), "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(value) for col, value in df.isna().sum().items()},
        "duplicate_records": int(df.duplicated().sum()),
        "statistics": statistics,
        "class_distribution": {str(k): int(v) for k, v in df[TARGET].value_counts().items()},
        "correlation_matrix": {"columns": FEATURES, "values": df[FEATURES].corr().round(4).values.tolist()},
        "histograms": histograms, "boxplots": boxplots,
    }


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    edges, rows, ece = np.linspace(0.0, 1.0, bins + 1), [], 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        count = int(mask.sum())
        if not count:
            rows.append({"lower": float(lower), "upper": float(upper), "count": 0,
                         "mean_confidence": None, "accuracy": None})
            continue
        bin_confidence, bin_accuracy = float(confidence[mask].mean()), float(correct[mask].mean())
        ece += (count / len(y_true)) * abs(bin_accuracy - bin_confidence)
        rows.append({"lower": round(float(lower), 3), "upper": round(float(upper), 3),
                     "count": count, "mean_confidence": round(bin_confidence, 6),
                     "accuracy": round(bin_accuracy, 6)})
    return float(ece), rows


def bootstrap_interval(y_true: np.ndarray, y_pred: np.ndarray, metric, samples: int = 1000) -> list[float]:
    rng, values = np.random.default_rng(RANDOM_STATE), []
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), len(y_true))
        values.append(float(metric(y_true[indices], y_pred[indices])))
    return [round(float(v), 6) for v in np.percentile(values, [2.5, 97.5])]


def make_ood_profile(x_train: pd.DataFrame) -> dict:
    mean = x_train.mean().to_numpy()
    inverse = np.linalg.pinv(np.cov(x_train.to_numpy(), rowvar=False))
    centered = x_train.to_numpy() - mean
    distances = np.sqrt(np.einsum("ij,jk,ik->i", centered, inverse, centered))
    ranges = {}
    for feature in FEATURES:
        series = x_train[feature]
        ranges[feature] = {
            "min": float(series.min()), "max": float(series.max()),
            "q01": float(series.quantile(0.01)), "q99": float(series.quantile(0.99)),
            "mean": float(series.mean()), "std": float(series.std()),
        }
    return {
        "method": "Mahalanobis distance on the unscaled training partition", "features": FEATURES,
        "mean": mean.tolist(), "inverse_covariance": inverse.tolist(),
        "warning_threshold": float(np.quantile(distances, 0.95)),
        "abstain_threshold": float(np.quantile(distances, 0.995)), "ranges": ranges,
    }


def main() -> None:
    print("Loading and validating dataset...")
    df = pd.read_csv(DATASET_PATH)
    if missing := set(FEATURES + [TARGET]).difference(df.columns):
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Dataset contains missing values; define an explicit imputation policy first.")
    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in FEATURES):
        raise TypeError("All input features must be numeric.")
    write_json(MODELS_DIR / "eda.json", build_eda(df))

    encoder = LabelEncoder()
    y, x = encoder.fit_transform(df[TARGET]), df[FEATURES].copy()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "KNN": (Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier())]),
                {"model__n_neighbors": [3, 5, 7], "model__weights": ["uniform", "distance"],
                 "model__metric": ["euclidean", "manhattan"]}),
        "Decision Tree": (Pipeline([("model", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
                          {"model__criterion": ["gini", "entropy"], "model__max_depth": [None, 10, 20],
                           "model__min_samples_split": [2, 5], "model__min_samples_leaf": [1, 2]}),
        "Random Forest": (Pipeline([("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))]),
                          {"model__n_estimators": [100, 250], "model__max_depth": [None, 10, 20],
                           "model__min_samples_split": [2, 5], "model__min_samples_leaf": [1, 2]}),
        "SVM": (Pipeline([("scale", StandardScaler()), ("model", SVC(random_state=RANDOM_STATE))]),
                {"model__C": [0.1, 1, 10], "model__gamma": ["scale", "auto"],
                 "model__kernel": ["rbf", "linear"]}),
    }
    scoring = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy", "accuracy": "accuracy"}
    searches, comparison = {}, []
    for name, (pipeline, parameters) in candidates.items():
        print(f"Cross-validating {name}...")
        search = GridSearchCV(pipeline, parameters, scoring=scoring, refit="macro_f1", cv=cv,
                              n_jobs=-1, return_train_score=False)
        search.fit(x_train, y_train)
        index, searches[name] = search.best_index_, search
        comparison.append({
            "model": name,
            "cv_macro_f1": round(float(search.cv_results_["mean_test_macro_f1"][index]), 6),
            "cv_macro_f1_std": round(float(search.cv_results_["std_test_macro_f1"][index]), 6),
            "cv_balanced_accuracy": round(float(search.cv_results_["mean_test_balanced_accuracy"][index]), 6),
            "cv_accuracy": round(float(search.cv_results_["mean_test_accuracy"][index]), 6),
            "best_params": search.best_params_,
        })

    baseline_scores = cross_validate(Pipeline([("model", DummyClassifier(strategy="most_frequent"))]),
                                     x_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
    comparison.append({
        "model": "Dummy baseline", "cv_macro_f1": round(float(baseline_scores["test_macro_f1"].mean()), 6),
        "cv_macro_f1_std": round(float(baseline_scores["test_macro_f1"].std()), 6),
        "cv_balanced_accuracy": round(float(baseline_scores["test_balanced_accuracy"].mean()), 6),
        "cv_accuracy": round(float(baseline_scores["test_accuracy"].mean()), 6),
        "best_params": {"strategy": "most_frequent"},
    })
    champion_name = max(comparison[:-1], key=lambda row: row["cv_macro_f1"])["model"]
    print(f"Selected {champion_name} using training-only macro F1.")
    champion = CalibratedClassifierCV(estimator=clone(searches[champion_name].best_estimator_),
                                      method="temperature", cv=cv, ensemble=False)
    champion.fit(x_train, y_train)
    probabilities, predictions = champion.predict_proba(x_test), champion.predict(x_test)
    labels, one_hot = np.arange(len(encoder.classes_)), np.eye(len(encoder.classes_))[y_test]
    ece, reliability = expected_calibration_error(y_test, probabilities)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "accuracy_ci_95": bootstrap_interval(y_test, predictions, accuracy_score),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, predictions)), 6),
        "macro_precision": round(float(precision_score(y_test, predictions, average="macro", zero_division=0)), 6),
        "macro_recall": round(float(recall_score(y_test, predictions, average="macro", zero_division=0)), 6),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro", zero_division=0)), 6),
        "macro_f1_ci_95": bootstrap_interval(y_test, predictions,
            lambda truth, pred: f1_score(truth, pred, average="macro", zero_division=0)),
        "weighted_f1": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 6),
        "top_3_accuracy": round(float(top_k_accuracy_score(y_test, probabilities, k=3, labels=labels)), 6),
        "top_5_accuracy": round(float(top_k_accuracy_score(y_test, probabilities, k=5, labels=labels)), 6),
        "log_loss": round(float(log_loss(y_test, probabilities, labels=labels)), 6),
        "multiclass_brier": round(float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))), 6),
        "expected_calibration_error": round(ece, 6),
    }
    report = classification_report(y_test, predictions, labels=labels, target_names=encoder.classes_,
                                   output_dict=True, zero_division=0)
    per_class = [{"crop": crop, **{key: round(float(report[crop][key]), 6)
                  for key in ("precision", "recall", "f1-score", "support")}} for crop in encoder.classes_]

    print("Calculating model-agnostic permutation importance...")
    importance = permutation_importance(champion, x_test, y_test, scoring="f1_macro", n_repeats=20,
                                        random_state=RANDOM_STATE, n_jobs=-1)
    importance_rows = sorted(
        [{"feature": feature, "mean": round(float(mean), 6), "std": round(float(std), 6)}
         for feature, mean, std in zip(FEATURES, importance.importances_mean, importance.importances_std)],
        key=lambda row: row["mean"], reverse=True,
    )

    model_path, encoder_path = MODELS_DIR / "champion_pipeline.pkl", MODELS_DIR / "label_encoder.pkl"
    joblib.dump(champion, model_path, compress=3)
    joblib.dump(encoder, encoder_path, compress=3)
    write_json(MODELS_DIR / "ood_profile.json", make_ood_profile(x_train))
    train_profiles = x_train.copy()
    train_profiles[TARGET] = encoder.inverse_transform(y_train)
    crop_profiles = {crop: {feature: {"mean": float(group[feature].mean()),
                                      "std": float(group[feature].std())} for feature in FEATURES}
                     for crop, group in train_profiles.groupby(TARGET)}
    write_json(MODELS_DIR / "crop_profiles.json", crop_profiles)
    write_json(MODELS_DIR / "feature_importance.json", {
        "method": "permutation importance on the untouched test partition using macro F1", "rows": importance_rows})
    comparison.sort(key=lambda row: row["cv_macro_f1"], reverse=True)
    model_report = {
        "selection_policy": "Highest 5-fold macro F1 on the training partition; test set not used for selection.",
        "selection_metric": "macro_f1", "models": comparison,
        "champion": {"model": champion_name,
                     "calibration": "temperature scaling with internal 5-fold cross-validation",
                     "test_metrics": metrics,
                     "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
                     "per_class": per_class, "reliability": reliability},
        "label_classes": encoder.classes_.tolist(), "train_records": int(len(x_train)),
        "test_records": int(len(x_test)), "test_size": TEST_SIZE,
    }
    write_json(MODELS_DIR / "model_comparison.json", model_report)
    manifest = {
        "schema_version": 2, "project": "Kshetra Sense", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": model_path.name, "encoder_file": encoder_path.name,
        "model_sha256": sha256(model_path), "encoder_sha256": sha256(encoder_path),
        "dataset_file": str(DATASET_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "dataset_sha256": sha256(DATASET_PATH), "champion_model": champion_name,
        "feature_order": FEATURES, "target": TARGET, "classes": encoder.classes_.tolist(),
        "confidence_threshold": CONFIDENCE_THRESHOLD, "ood_profile_file": "ood_profile.json",
        "crop_profiles_file": "crop_profiles.json",
        "validation_scope": "Internal stratified holdout only; no geographic, seasonal, or prospective validation.",
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__}, "test_metrics": metrics,
    }
    write_json(MODELS_DIR / "manifest.json", manifest)
    print(f"Done. {champion_name} test macro F1={metrics['macro_f1']:.4f}; accuracy={metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
