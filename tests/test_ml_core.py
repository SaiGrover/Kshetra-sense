from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_core import (
    assess_ood,
    crop_fit,
    load_model_bundle,
    local_sensitivity,
    predict_ranked,
    validate_external_dataframe,
)


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


@pytest.fixture(scope="module")
def bundle():
    return load_model_bundle(ROOT / "models")


def test_artifact_contract_and_integrity(bundle):
    assert bundle.manifest["schema_version"] == 2
    assert bundle.manifest["feature_order"] == FEATURES
    assert len(bundle.manifest["classes"]) == 22
    assert not bundle.warnings


def test_ranked_probabilities_are_ordered(bundle):
    values = [90, 42, 43, 21.0, 82, 6.5, 203]
    ranked = predict_ranked(bundle, values, top_k=5)
    probabilities = [row["probability"] for row in ranked]
    assert len(ranked) == 5
    assert all(0 <= value <= 1 for value in probabilities)
    assert probabilities == sorted(probabilities, reverse=True)
    full = bundle.model.predict_proba(pd.DataFrame([values], columns=FEATURES))[0]
    assert np.isclose(full.sum(), 1.0)


def test_ood_guardrails(bundle):
    typical = [bundle.ood_profile["ranges"][feature]["mean"] for feature in FEATURES]
    assert assess_ood(typical, bundle.ood_profile)["status"] == "in_distribution"
    extreme = [1000.0] * len(FEATURES)
    assessment = assess_ood(extreme, bundle.ood_profile)
    assert assessment["status"] == "abstain"
    assert set(assessment["outside_features"]) == set(FEATURES)


def test_explanations_are_complete(bundle):
    values = [90, 42, 43, 21.0, 82, 6.5, 203]
    crop = predict_ranked(bundle, values, top_k=1)[0]["crop"]
    sensitivity = local_sensitivity(bundle, values)
    fit = crop_fit(bundle, crop, values)
    assert set(sensitivity["feature"]) == set(FEATURES)
    assert set(fit["feature"]) == set(FEATURES)
    assert np.isfinite(sensitivity["probability_change"]).all()


def test_external_validation_schema_and_metrics(bundle):
    external = pd.read_csv(ROOT / "dataset" / "Crop_recommendation.csv").groupby("label").head(4)
    result = validate_external_dataframe(bundle, external)
    assert result["records"] == 88
    assert 0 <= result["macro_f1"] <= 1
    assert len(result["predictions"]) == 88
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_external_dataframe(bundle, external.drop(columns=["label"]))
