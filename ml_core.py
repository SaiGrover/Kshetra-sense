"""Framework-independent inference, validation, and safety utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, log_loss,
    precision_score, recall_score, top_k_accuracy_score,
)


@dataclass(frozen=True)
class ModelBundle:
    model: object
    encoder: object
    manifest: dict
    ood_profile: dict
    crop_profiles: dict
    warnings: tuple[str, ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model_bundle(models_dir: str | Path) -> ModelBundle:
    """Load artifacts and reject incompatible or silently modified packages."""
    root = Path(models_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise RuntimeError("Unsupported artifact schema. Retrain the project before serving it.")
    model_path, encoder_path = root / manifest["model_file"], root / manifest["encoder_file"]
    for path, key in ((model_path, "model_sha256"), (encoder_path, "encoder_sha256")):
        if file_sha256(path) != manifest[key]:
            raise RuntimeError(f"Artifact integrity check failed for {path.name}.")
    warnings = []
    trained_version = manifest.get("versions", {}).get("scikit_learn")
    if trained_version and trained_version != sklearn.__version__:
        warnings.append(f"Model trained with scikit-learn {trained_version}; runtime is {sklearn.__version__}.")
    return ModelBundle(
        model=joblib.load(model_path), encoder=joblib.load(encoder_path), manifest=manifest,
        ood_profile=json.loads((root / manifest["ood_profile_file"]).read_text(encoding="utf-8")),
        crop_profiles=json.loads((root / manifest["crop_profiles_file"]).read_text(encoding="utf-8")),
        warnings=tuple(warnings),
    )


def values_frame(values: Mapping[str, float] | Sequence[float], features: Sequence[str]) -> pd.DataFrame:
    if isinstance(values, Mapping):
        row = [float(values[name]) for name in features]
    else:
        row = [float(value) for value in values]
    if len(row) != len(features) or not np.isfinite(row).all():
        raise ValueError("Provide one finite numeric value for every feature.")
    return pd.DataFrame([row], columns=list(features))


def assess_ood(values: Mapping[str, float] | Sequence[float], profile: dict) -> dict:
    """Flag novel inputs using observed ranges and multivariate distance."""
    features = profile["features"]
    row = values_frame(values, features).iloc[0]
    outside = []
    unusual = []
    for feature in features:
        bounds = profile["ranges"][feature]
        if row[feature] < bounds["min"] or row[feature] > bounds["max"]:
            outside.append(feature)
        elif row[feature] < bounds["q01"] or row[feature] > bounds["q99"]:
            unusual.append(feature)
    delta = row.to_numpy(dtype=float) - np.asarray(profile["mean"], dtype=float)
    distance = float(np.sqrt(max(0.0, delta @ np.asarray(profile["inverse_covariance"]) @ delta)))
    if outside or distance > profile["abstain_threshold"]:
        status = "abstain"
    elif unusual or distance > profile["warning_threshold"]:
        status = "warning"
    else:
        status = "in_distribution"
    return {
        "status": status, "distance": distance,
        "warning_threshold": float(profile["warning_threshold"]),
        "abstain_threshold": float(profile["abstain_threshold"]),
        "outside_features": outside, "unusual_features": unusual,
    }


def predict_ranked(bundle: ModelBundle, values: Mapping[str, float] | Sequence[float], top_k: int = 3) -> list[dict]:
    frame = values_frame(values, bundle.manifest["feature_order"])
    probabilities = bundle.model.predict_proba(frame)[0]
    order = np.argsort(probabilities)[::-1][:top_k]
    labels = bundle.encoder.inverse_transform(order)
    return [{"crop": str(label), "probability": float(probabilities[index]), "class_index": int(index)}
            for label, index in zip(labels, order)]


def local_sensitivity(bundle: ModelBundle, values: Mapping[str, float] | Sequence[float]) -> pd.DataFrame:
    """One-at-a-time sensitivity; useful context, not a causal explanation."""
    features = bundle.manifest["feature_order"]
    frame = values_frame(values, features)
    base_probabilities = bundle.model.predict_proba(frame)[0]
    winner = int(base_probabilities.argmax())
    rows = []
    for feature in features:
        modified = frame.copy()
        training_mean = float(bundle.ood_profile["ranges"][feature]["mean"])
        modified.loc[0, feature] = training_mean
        changed = float(bundle.model.predict_proba(modified)[0, winner])
        rows.append({
            "feature": feature, "input_value": float(frame.loc[0, feature]),
            "reference_value": training_mean,
            "probability_change": float(base_probabilities[winner] - changed),
        })
    return pd.DataFrame(rows).sort_values("probability_change", key=lambda col: col.abs(), ascending=False)


def crop_fit(bundle: ModelBundle, crop: str, values: Mapping[str, float] | Sequence[float]) -> pd.DataFrame:
    features = bundle.manifest["feature_order"]
    row = values_frame(values, features).iloc[0]
    profile = bundle.crop_profiles[crop]
    rows = []
    for feature in features:
        std = profile[feature]["std"] or 1.0
        rows.append({"feature": feature, "value": float(row[feature]),
                     "crop_average": float(profile[feature]["mean"]),
                     "standard_deviations": float((row[feature] - profile[feature]["mean"]) / std)})
    return pd.DataFrame(rows)


def validate_external_dataframe(bundle: ModelBundle, dataframe: pd.DataFrame, target: str = "label") -> dict:
    """Evaluate a labelled CSV without changing the deployed model."""
    features = bundle.manifest["feature_order"]
    missing = [column for column in [*features, target] if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    frame = dataframe[[*features, target]].copy()
    if frame.isna().any().any():
        raise ValueError("External data contains missing values.")
    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="raise")
    unknown = sorted(set(frame[target].astype(str)) - set(bundle.encoder.classes_))
    if unknown:
        raise ValueError(f"Unknown crop labels: {', '.join(unknown)}")
    y_true = bundle.encoder.transform(frame[target].astype(str))
    probabilities = bundle.model.predict_proba(frame[features])
    y_pred = probabilities.argmax(axis=1)
    labels = np.arange(len(bundle.encoder.classes_))
    ood = [assess_ood(row, bundle.ood_profile)["status"] for row in frame[features].to_dict("records")]
    return {
        "records": int(len(frame)), "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "top_3_accuracy": float(top_k_accuracy_score(y_true, probabilities, k=3, labels=labels)),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "ood_warning_rate": float(np.mean(np.asarray(ood) != "in_distribution")),
        "predictions": bundle.encoder.inverse_transform(y_pred).tolist(),
    }
