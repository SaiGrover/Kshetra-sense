"""
Crop Recommendation System - Model Training Script
Trains KNN, Decision Tree, Random Forest, SVM with GridSearchCV
Saves all models and analysis artifacts to JSON for the API
"""

import pandas as pd
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTE
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "Crop_recommendation.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# ── EDA / Dataset Analysis ──────────────────────────────────────────────────
features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
target = "label"

eda = {
    "shape": list(df.shape),
    "columns": list(df.columns),
    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    "missing_values": df.isnull().sum().to_dict(),
    "duplicate_records": int(df.duplicated().sum()),
    "statistics": {},
    "class_distribution": df[target].value_counts().to_dict(),
}

for col in features:
    eda["statistics"][col] = {
        "mean":   round(float(df[col].mean()), 4),
        "median": round(float(df[col].median()), 4),
        "std":    round(float(df[col].std()), 4),
        "min":    round(float(df[col].min()), 4),
        "max":    round(float(df[col].max()), 4),
        "q25":    round(float(df[col].quantile(0.25)), 4),
        "q75":    round(float(df[col].quantile(0.75)), 4),
    }

# Correlation matrix
corr = df[features].corr().round(4)
eda["correlation_matrix"] = {
    "columns": features,
    "values": corr.values.tolist()
}

# Per-feature histograms (bins of 20)
hist_data = {}
for col in features:
    counts, bin_edges = np.histogram(df[col], bins=20)
    hist_data[col] = {
        "counts": counts.tolist(),
        "bin_edges": [round(float(e), 4) for e in bin_edges.tolist()],
    }
eda["histograms"] = hist_data

# Boxplot stats per feature
boxplot_data = {}
for col in features:
    boxplot_data[col] = {
        "min":    round(float(df[col].min()), 4),
        "q25":    round(float(df[col].quantile(0.25)), 4),
        "median": round(float(df[col].median()), 4),
        "q75":    round(float(df[col].quantile(0.75)), 4),
        "max":    round(float(df[col].max()), 4),
    }
eda["boxplots"] = boxplot_data

with (MODELS_DIR / "eda.json").open("w", encoding="utf-8") as f:
    json.dump(eda, f)
print("EDA saved.")

# ── Preprocessing ────────────────────────────────────────────────────────────
le = LabelEncoder()
df["label_enc"] = le.fit_transform(df[target])
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

X = df[features].values
y = df["label_enc"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Check class balance
class_counts = np.bincount(y)
is_imbalanced = (max(class_counts) / min(class_counts)) > 2
smote_applied = False
if is_imbalanced:
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    smote_applied = True
    print("SMOTE applied.")

# ── Model Definitions + GridSearchCV ────────────────────────────────────────
param_grids = {
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        }
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "kernel": ["rbf", "linear"],
        }
    }
}

model_file_map = {
    "KNN": "best_knn.pkl",
    "Decision Tree": "best_dt.pkl",
    "Random Forest": "best_rf.pkl",
    "SVM": "best_svm.pkl",
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
comparison_results = []

for name, cfg in param_grids.items():
    print(f"Training {name}...")
    gs = GridSearchCV(cfg["model"], cfg["params"], cv=cv,
                      scoring="accuracy", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cv_scores = cross_val_score(best, X_scaled, y, cv=cv, scoring="accuracy")
    cm = confusion_matrix(y_test, y_pred).tolist()

    result = {
        "model": name,
        "accuracy":   round(float(acc), 4),
        "precision":  round(float(prec), 4),
        "recall":     round(float(rec), 4),
        "f1_score":   round(float(f1), 4),
        "cv_mean":    round(float(cv_scores.mean()), 4),
        "cv_std":     round(float(cv_scores.std()), 4),
        "best_params": gs.best_params_,
        "confusion_matrix": cm,
    }
    comparison_results.append(result)
    joblib.dump(best, MODELS_DIR / model_file_map[name])
    print(f"  {name} accuracy: {acc:.4f}")

# Sort by accuracy
comparison_results.sort(key=lambda x: x["accuracy"], reverse=True)

with (MODELS_DIR / "model_comparison.json").open("w", encoding="utf-8") as f:
    json.dump({
        "models": comparison_results,
        "smote_applied": smote_applied,
        "label_classes": list(le.classes_),
        "test_size": 0.2,
    }, f)
print("Model comparison saved.")

# ── Feature Importance ───────────────────────────────────────────────────────
fi_data = {}
for model_name, pkl_name in [("Decision Tree", "best_dt.pkl"), ("Random Forest", "best_rf.pkl")]:
    model = joblib.load(MODELS_DIR / pkl_name)
    importances = model.feature_importances_
    fi_data[model_name] = {
        f: round(float(v), 6)
        for f, v in sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    }

with (MODELS_DIR / "feature_importance.json").open("w", encoding="utf-8") as f:
    json.dump(fi_data, f)
print("Feature importance saved.")

print("\n✅ All models trained and saved successfully!")
best_model_info = comparison_results[0]
print(f"Best model: {best_model_info['model']} with accuracy {best_model_info['accuracy']}")
