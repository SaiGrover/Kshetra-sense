"""CropLens: deployable Streamlit crop recommendation application."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CROP_EMOJI = {
    "apple": "🍎", "banana": "🍌", "blackgram": "🫘", "chickpea": "🫘",
    "coconut": "🥥", "coffee": "☕", "cotton": "🌸", "grapes": "🍇",
    "jute": "🌿", "kidneybeans": "🫘", "lentil": "🫘", "maize": "🌽",
    "mango": "🥭", "mothbeans": "🫘", "mungbean": "🫘", "muskmelon": "🍈",
    "orange": "🍊", "papaya": "🍈", "pigeonpeas": "🌿",
    "pomegranate": "🍎", "rice": "🌾", "watermelon": "🍉",
}


st.set_page_config(
    page_title="CropLens | Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    .stApp { background: #f7faf5; }
    [data-testid="stSidebar"] { background: #102318; }
    [data-testid="stSidebar"] * { color: #f2f7f0; }
    .hero {
        padding: 2.2rem 2.5rem; border-radius: 24px; margin-bottom: 1.4rem;
        color: #f7fff4; background: linear-gradient(120deg, #123c22, #23733e);
        box-shadow: 0 18px 50px rgba(18, 60, 34, .18);
    }
    .hero h1 { font-size: clamp(2.25rem, 5vw, 4.2rem); margin: 0; letter-spacing: -.04em; }
    .hero p { max-width: 760px; color: #d4ead8; font-size: 1.05rem; margin: .8rem 0 0; }
    .result-card {
        padding: 1.5rem; border-radius: 18px; border: 1px solid #b9ddc1;
        background: linear-gradient(135deg, #eef9ef, #ffffff); text-align: center;
    }
    .result-crop { color: #145c2d; font-size: 2rem; font-weight: 800; text-transform: capitalize; }
    .eyebrow { color: #397a4b; font-size: .78rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #dce9dc; padding: 1rem; border-radius: 16px; }
    div[data-testid="stForm"] { background: white; border: 1px solid #dce9dc; padding: 1.2rem; border-radius: 18px; }
    .small-note { color: #607064; font-size: .86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading recommendation model…")
def load_artifacts() -> tuple[object, object, object, dict, dict, dict]:
    """Load immutable model artifacts once per Streamlit process."""
    required = {
        "scaler": MODELS_DIR / "scaler.pkl",
        "encoder": MODELS_DIR / "label_encoder.pkl",
        "model": MODELS_DIR / "best_rf.pkl",
        "eda": MODELS_DIR / "eda.json",
        "comparison": MODELS_DIR / "model_comparison.json",
        "importance": MODELS_DIR / "feature_importance.json",
    }
    missing = [path.name for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {', '.join(missing)}")

    with required["eda"].open(encoding="utf-8") as handle:
        eda = json.load(handle)
    with required["comparison"].open(encoding="utf-8") as handle:
        comparison = json.load(handle)
    with required["importance"].open(encoding="utf-8") as handle:
        importance = json.load(handle)

    return (
        joblib.load(required["scaler"]),
        joblib.load(required["encoder"]),
        joblib.load(required["model"]),
        eda,
        comparison,
        importance,
    )


def predict_crops(values: list[float], scaler: object, encoder: object, model: object) -> pd.DataFrame:
    """Return the five most likely crops in descending order."""
    feature_array = np.asarray([values], dtype=float)
    probabilities = model.predict_proba(scaler.transform(feature_array))[0]
    indices = np.argsort(probabilities)[::-1][:5]
    labels = encoder.classes_
    return pd.DataFrame(
        {
            "Crop": [str(labels[index]).replace("beans", " beans").title() for index in indices],
            "Confidence": [float(probabilities[index]) for index in indices],
        }
    )


try:
    scaler, encoder, model, eda, comparison, importance = load_artifacts()
except Exception as exc:
    st.error("CropLens could not load its trained model artifacts.")
    st.exception(exc)
    st.stop()


with st.sidebar:
    st.markdown("## 🌱 CropLens")
    st.caption("Data-guided crop recommendation")
    page = st.radio(
        "Navigate",
        ["Recommend", "Overview", "Model performance", "Data explorer"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Random Forest · 2,200 samples · 22 crop classes")
    st.caption("Use recommendations as decision support alongside local agronomic advice.")


if page == "Recommend":
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow" style="color:#aee8ba">Machine learning for practical farming</div>
          <h1>Match your field to the right crop.</h1>
          <p>Enter a soil test and recent climate conditions. CropLens ranks the five crops most compatible with those measurements.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    form_col, result_col = st.columns([1.08, 0.92], gap="large")
    with form_col:
        st.subheader("Field conditions")
        st.caption("Use laboratory soil-test values where available. The sample preset represents rice-like conditions.")
        with st.form("prediction_form"):
            left, right = st.columns(2)
            with left:
                nitrogen = st.number_input("Nitrogen (N), mg/kg", 0.0, 200.0, 90.0, 1.0)
                phosphorus = st.number_input("Phosphorus (P), mg/kg", 0.0, 200.0, 42.0, 1.0)
                potassium = st.number_input("Potassium (K), mg/kg", 0.0, 250.0, 43.0, 1.0)
                soil_ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, 0.1)
            with right:
                temperature = st.number_input("Temperature, °C", -10.0, 60.0, 21.0, 0.5)
                humidity = st.number_input("Humidity, %", 0.0, 100.0, 82.0, 1.0)
                rainfall = st.number_input("Rainfall, mm", 0.0, 500.0, 203.0, 1.0)
            submitted = st.form_submit_button("Recommend crops", type="primary", use_container_width=True)

    with result_col:
        st.subheader("Recommendation")
        if submitted:
            values = [nitrogen, phosphorus, potassium, temperature, humidity, soil_ph, rainfall]
            ranked = predict_crops(values, scaler, encoder, model)
            best_crop = ranked.iloc[0]["Crop"]
            best_key = str(best_crop).lower().replace(" ", "")
            best_confidence = float(ranked.iloc[0]["Confidence"])
            st.markdown(
                f"""
                <div class="result-card">
                  <div style="font-size:3.2rem">{CROP_EMOJI.get(best_key, '🌿')}</div>
                  <div class="eyebrow">Best match</div>
                  <div class="result-crop">{best_crop}</div>
                  <div style="color:#397a4b;font-weight:700">{best_confidence:.1%} model confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            chart = ranked.set_index("Crop")
            st.bar_chart(chart, horizontal=True, color="#2f8f4e")
            st.caption("Confidence reflects the model's relative certainty, not guaranteed yield or profitability.")
        else:
            st.info("Complete the form and select **Recommend crops** to see ranked results.")

elif page == "Overview":
    st.title("Project overview")
    st.caption("A compact view of the training data and recommendation pipeline.")
    best_model = comparison["models"][0]
    metrics = st.columns(4)
    metrics[0].metric("Training samples", f"{eda['shape'][0]:,}")
    metrics[1].metric("Crop classes", len(eda["class_distribution"]))
    metrics[2].metric("Best model", best_model["model"])
    metrics[3].metric("Test accuracy", f"{best_model['accuracy']:.2%}")

    st.subheader("Balanced class coverage")
    distribution = pd.Series(eda["class_distribution"], name="Samples").sort_index()
    st.bar_chart(distribution, color="#2f8f4e")

    st.subheader("How a recommendation is produced")
    st.markdown(
        "**Soil & climate inputs** → standardize seven features → Random Forest probabilities → ranked crop matches"
    )
    st.info("The model was evaluated on a held-out 20% test split. Real-world performance can differ by region, season, and measurement quality.")

elif page == "Model performance":
    st.title("Model performance")
    st.caption("Four tuned classifiers evaluated on the same held-out test set.")
    models = pd.DataFrame(comparison["models"])
    display = models[["model", "accuracy", "precision", "recall", "f1_score", "cv_mean", "cv_std"]].copy()
    display.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 score", "CV mean", "CV std"]
    st.dataframe(
        display.style.format({column: "{:.2%}" for column in display.columns if column != "Model"}),
        hide_index=True,
        use_container_width=True,
    )
    st.subheader("Accuracy comparison")
    st.bar_chart(display.set_index("Model")["Accuracy"], horizontal=True, color="#2f8f4e")

    st.subheader("Random Forest feature importance")
    feature_importance = pd.Series(importance["Random Forest"], name="Importance").sort_values()
    st.bar_chart(feature_importance, horizontal=True, color="#d39c36")
    st.caption("Feature importance shows model reliance, not causal effect on crop growth.")

elif page == "Data explorer":
    st.title("Data explorer")
    st.caption("Distribution and summary statistics for the seven model inputs.")
    feature = st.selectbox("Feature", FEATURES, format_func=lambda value: value.replace("ph", "pH").title())
    histogram = eda["histograms"][feature]
    edges = histogram["bin_edges"]
    histogram_frame = pd.DataFrame(
        {
            "Range midpoint": [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)],
            "Samples": histogram["counts"],
        }
    ).set_index("Range midpoint")
    st.bar_chart(histogram_frame, color="#2f8f4e")

    st.subheader("Feature summary")
    summary = pd.DataFrame(eda["statistics"]).T
    summary.index.name = "Feature"
    st.dataframe(summary, use_container_width=True)

    st.subheader("Correlation matrix")
    correlation = eda["correlation_matrix"]
    correlation_frame = pd.DataFrame(
        correlation["values"], index=correlation["columns"], columns=correlation["columns"]
    )
    st.dataframe(
        correlation_frame.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.2f}"),
        use_container_width=True,
    )
