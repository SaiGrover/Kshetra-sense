"""KshetraSense: an interactive crop recommendation dashboard."""

from __future__ import annotations

import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FEATURE_LABELS = {
    "N": "Nitrogen",
    "P": "Phosphorus",
    "K": "Potassium",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "Soil pH",
    "rainfall": "Rainfall",
}
CROP_ICONS = {
    "apple": "AP", "banana": "BN", "blackgram": "BG", "chickpea": "CP",
    "coconut": "CO", "coffee": "CF", "cotton": "CT", "grapes": "GR",
    "jute": "JT", "kidneybeans": "KB", "lentil": "LN", "maize": "MZ",
    "mango": "MG", "mothbeans": "MB", "mungbean": "MU", "muskmelon": "MM",
    "orange": "OR", "papaya": "PA", "pigeonpeas": "PP", "pomegranate": "PG",
    "rice": "RI", "watermelon": "WM",
}


st.set_page_config(
    page_title="KshetraSense | Crop Intelligence",
    page_icon="KS",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

    :root {
        --bg: #080d0a;
        --surface: #101713;
        --surface-2: #151f18;
        --surface-3: #1a271e;
        --line: #26352b;
        --line-soft: rgba(161, 205, 170, .12);
        --text: #edf5ef;
        --muted: #8fa398;
        --green: #7ee787;
        --lime: #c8f169;
        --amber: #f0c75e;
        --danger: #ff7b72;
    }

    @keyframes rise {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes glow {
        0%, 100% { opacity: .55; transform: scale(1); }
        50% { opacity: .8; transform: scale(1.05); }
    }
    @keyframes sweep {
        from { background-position: 0% 50%; }
        to { background-position: 200% 50%; }
    }

    html, body, [class*="css"], .stApp {
        font-family: 'DM Sans', sans-serif;
        color: var(--text);
    }
    .stApp,
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 72% -10%, rgba(126, 231, 135, .07), transparent 30rem),
            var(--bg);
    }
    .main .block-container {
        max-width: 1440px;
        padding: 1.5rem 2.4rem 4rem;
    }
    [data-testid="stHeader"] {
        background: rgba(8, 13, 10, .8);
        backdrop-filter: blur(14px);
        border-bottom: 1px solid rgba(255,255,255,.04);
    }
    [data-testid="stToolbar"] { visibility: hidden; }
    footer { display: none; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a110c 0%, #0c140f 55%, #080d0a 100%);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 1.55rem 1rem 1.25rem;
    }
    .brand-lockup {
        padding: .4rem .55rem 1.15rem;
        border-bottom: 1px solid var(--line-soft);
        margin-bottom: 1rem;
    }
    .brand-row { display: flex; align-items: center; gap: .75rem; }
    .brand-mark {
        width: 38px; height: 38px; border-radius: 12px;
        display: grid; place-items: center;
        background: linear-gradient(145deg, rgba(126,231,135,.22), rgba(200,241,105,.06));
        border: 1px solid rgba(126,231,135,.25);
        color: var(--green); font-family: 'Space Grotesk'; font-weight: 700;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 0 28px rgba(126,231,135,.08);
    }
    .brand-name { font-family: 'Space Grotesk'; font-size: 1rem; font-weight: 700; letter-spacing: -.02em; }
    .brand-sub { color: var(--muted); font-size: .68rem; margin-top: .08rem; }
    .nav-label, .mono-label {
        color: #6f8477; font: 600 .64rem 'JetBrains Mono', monospace;
        letter-spacing: .12em; text-transform: uppercase;
    }
    .nav-label { padding: .1rem .55rem .4rem; }
    [data-testid="stSidebar"] [data-baseweb="radio"] > div { gap: .24rem; }
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        min-height: 42px; padding: .55rem .72rem; border-radius: 10px;
        color: var(--muted); font-size: .85rem; font-weight: 600;
        border: 1px solid transparent; transition: .18s ease;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
        color: var(--text); background: rgba(126,231,135,.06);
        border-color: rgba(126,231,135,.09);
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] label:has(input:checked) {
        color: var(--green); background: rgba(126,231,135,.09);
        border-color: rgba(126,231,135,.16);
        box-shadow: inset 3px 0 0 var(--green);
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] label > div:first-child { display: none; }
    .sidebar-status {
        margin-top: 1.2rem; padding: .9rem; border-radius: 12px;
        background: rgba(255,255,255,.025); border: 1px solid var(--line-soft);
    }
    .status-row { display:flex; align-items:center; gap:.5rem; font-size:.74rem; color:#a7b9ae; }
    .status-dot { width:7px; height:7px; border-radius:50%; background:var(--green); box-shadow:0 0 10px var(--green); }
    .sidebar-note { color:#607267; font-size:.67rem; line-height:1.5; margin-top:.7rem; }

    /* Shared content */
    .topline {
        display:flex; align-items:center; justify-content:space-between;
        margin-bottom: 1.15rem; animation: rise .45s ease both;
    }
    .topline-title { color: var(--muted); font-size: .78rem; }
    .live-pill {
        display:flex; align-items:center; gap:.45rem; padding:.38rem .65rem;
        color:#a8b9ae; border:1px solid var(--line); border-radius:999px;
        background:rgba(255,255,255,.025); font:500 .66rem 'JetBrains Mono';
    }
    .hero {
        position:relative; overflow:hidden; min-height: 252px;
        display:flex; align-items:center; padding:2.7rem 3rem;
        border:1px solid #27372c; border-radius:24px;
        background: linear-gradient(125deg, #111b14 0%, #0d1811 56%, #14271a 100%);
        box-shadow: 0 24px 70px rgba(0,0,0,.27), inset 0 1px 0 rgba(255,255,255,.035);
        animation: rise .55s ease both;
    }
    .hero::before {
        content:''; position:absolute; width:420px; height:420px; border-radius:50%;
        right:-95px; top:-220px; background:rgba(126,231,135,.08); animation:glow 7s ease-in-out infinite;
    }
    .hero::after {
        content:''; position:absolute; width:370px; height:370px; border-radius:50%;
        right:70px; bottom:-310px; border:1px solid rgba(200,241,105,.11);
        box-shadow: 0 0 0 55px rgba(200,241,105,.018), 0 0 0 110px rgba(200,241,105,.012);
    }
    .hero-content { position:relative; z-index:1; max-width:790px; }
    .eyebrow { display:flex; align-items:center; gap:.6rem; margin-bottom:.8rem; color:var(--green); font:600 .68rem 'JetBrains Mono'; letter-spacing:.13em; text-transform:uppercase; }
    .eyebrow::before { content:''; width:22px; height:1px; background:var(--green); }
    .hero h1 {
        font:700 clamp(2.25rem,4.5vw,4.25rem)/.98 'Space Grotesk';
        letter-spacing:-.055em; margin:0; max-width:780px; color:var(--text);
    }
    .hero h1 span {
        background: linear-gradient(90deg, var(--green), var(--lime), var(--green));
        background-size:200% auto; animation:sweep 5s linear infinite;
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .hero p { max-width:670px; color:#9caf9f; font-size:.98rem; line-height:1.65; margin:1rem 0 0; }
    .hero-tags { display:flex; flex-wrap:wrap; gap:.48rem; margin-top:1.25rem; }
    .hero-tag { padding:.34rem .65rem; border:1px solid rgba(143,163,152,.18); border-radius:999px; background:rgba(255,255,255,.025); color:#91a598; font:500 .65rem 'JetBrains Mono'; }

    .metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin:1rem 0 1.45rem; }
    .metric-card {
        position:relative; overflow:hidden; min-height:112px; padding:1.05rem 1.15rem;
        background:linear-gradient(145deg,var(--surface),var(--surface-2));
        border:1px solid var(--line); border-radius:16px; animation:rise .5s ease both;
        transition:transform .2s ease,border-color .2s ease;
    }
    .metric-card:hover { transform:translateY(-3px); border-color:#3b5744; }
    .metric-card::before { content:''; position:absolute; inset:0 auto 0 0; width:2px; background:var(--tone,var(--green)); }
    .metric-kicker { color:var(--muted); font:600 .62rem 'JetBrains Mono'; letter-spacing:.09em; text-transform:uppercase; }
    .metric-value { color:var(--tone,var(--green)); font:700 1.75rem 'Space Grotesk'; letter-spacing:-.035em; margin-top:.38rem; }
    .metric-foot { color:#617367; font-size:.67rem; margin-top:.2rem; }

    .section-head { margin:1.7rem 0 1rem; }
    .section-head h2 { font:700 1.25rem 'Space Grotesk'; letter-spacing:-.025em; margin:0; }
    .section-head p { color:var(--muted); font-size:.8rem; margin:.28rem 0 0; }
    .panel {
        border:1px solid var(--line); border-radius:18px; padding:1.25rem;
        background:linear-gradient(145deg,rgba(16,23,19,.96),rgba(20,31,24,.88));
        box-shadow:inset 0 1px 0 rgba(255,255,255,.025); animation:rise .55s ease both;
    }
    .panel-title { font:700 1rem 'Space Grotesk'; letter-spacing:-.02em; }
    .panel-sub { color:var(--muted); font-size:.75rem; margin:.2rem 0 1rem; }

    /* Widgets */
    div[data-testid="stForm"] { border:0; padding:0; }
    div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label {
        color:#aabcaf !important; font-size:.76rem !important; font-weight:600 !important;
    }
    div[data-testid="stNumberInput"] input {
        color:var(--text); background:#0c130e; border-color:var(--line);
        font-family:'JetBrains Mono'; font-size:.78rem;
    }
    div[data-testid="stNumberInput"] > div > div,
    [data-baseweb="select"] > div {
        background:#0c130e !important; border-color:var(--line) !important; border-radius:9px !important;
    }
    div[data-testid="stNumberInput"] > div > div:focus-within,
    [data-baseweb="select"] > div:focus-within { border-color:var(--green) !important; box-shadow:0 0 0 2px rgba(126,231,135,.1) !important; }
    .stButton > button, [data-testid="stFormSubmitButton"] > button {
        min-height:44px; border-radius:10px; border:1px solid #98dd78;
        background:linear-gradient(135deg,#b6ed7a,#79d783); color:#0a150c;
        font-weight:800; box-shadow:0 8px 24px rgba(126,231,135,.12); transition:.18s ease;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
        color:#071109; border-color:#d8ff9c; transform:translateY(-1px); box-shadow:0 11px 30px rgba(126,231,135,.2);
    }
    [data-testid="stAlert"] { background:var(--surface) !important; border:1px solid var(--line) !important; border-radius:12px !important; color:var(--muted) !important; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    hr { border-color:var(--line-soft) !important; }

    /* Prediction */
    .result-shell {
        min-height:270px; position:relative; overflow:hidden; padding:1.55rem;
        border:1px solid rgba(126,231,135,.24); border-radius:18px;
        background:radial-gradient(circle at 80% 0%,rgba(126,231,135,.1),transparent 40%),linear-gradient(145deg,#111b14,#101712);
    }
    .result-shell::after { content:''; position:absolute; width:180px; height:180px; border:1px solid rgba(200,241,105,.1); border-radius:50%; right:-85px; bottom:-95px; }
    .crop-token { width:54px; height:54px; display:grid; place-items:center; border-radius:15px; background:linear-gradient(145deg,rgba(126,231,135,.19),rgba(200,241,105,.05)); border:1px solid rgba(126,231,135,.24); color:var(--green); font:700 .8rem 'JetBrains Mono'; }
    .best-label { color:var(--green); font:600 .64rem 'JetBrains Mono'; letter-spacing:.12em; text-transform:uppercase; margin-top:1.05rem; }
    .crop-name { font:700 2.65rem 'Space Grotesk'; letter-spacing:-.05em; line-height:1; text-transform:capitalize; margin:.32rem 0 .55rem; }
    .confidence { display:inline-flex; align-items:center; gap:.45rem; padding:.34rem .6rem; border-radius:8px; background:rgba(126,231,135,.08); color:#b8eab7; font:600 .68rem 'JetBrains Mono'; }
    .rank-list { margin-top:1rem; }
    .rank-row { display:grid; grid-template-columns:26px 1fr 48px; align-items:center; gap:.65rem; margin:.65rem 0; }
    .rank-no { color:#5f7365; font:600 .63rem 'JetBrains Mono'; }
    .rank-name { color:#c5d2c8; font-size:.76rem; margin-bottom:.28rem; }
    .rank-pct { color:#91a598; font:500 .65rem 'JetBrains Mono'; text-align:right; }
    .rank-track { height:5px; border-radius:99px; background:#233027; overflow:hidden; }
    .rank-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,var(--green),var(--lime)); }
    .empty-result { min-height:270px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; border:1px dashed #2d4032; border-radius:18px; background:rgba(255,255,255,.015); }
    .empty-icon { width:52px;height:52px;border-radius:50%;display:grid;place-items:center;background:rgba(126,231,135,.07);border:1px solid rgba(126,231,135,.15);color:var(--green);font:700 .72rem 'JetBrains Mono'; }
    .empty-title { font:600 .95rem 'Space Grotesk'; margin-top:.85rem; }
    .empty-copy { color:var(--muted); font-size:.74rem; max-width:260px; line-height:1.5; margin-top:.3rem; }
    .flow { display:grid; grid-template-columns:repeat(4,1fr); gap:.7rem; }
    .flow-step { position:relative; min-height:126px; padding:1rem; border:1px solid var(--line); border-radius:14px; background:var(--surface); }
    .flow-num { color:var(--green); font:600 .62rem 'JetBrains Mono'; }
    .flow-title { font:700 .86rem 'Space Grotesk'; margin:.62rem 0 .3rem; }
    .flow-copy { color:var(--muted); font-size:.7rem; line-height:1.48; }
    .notice { border-left:2px solid var(--amber); padding:.85rem 1rem; border-radius:0 10px 10px 0; background:rgba(240,199,94,.055); color:#a9aa92; font-size:.73rem; line-height:1.55; }

    @media (max-width: 900px) {
        .main .block-container { padding:1rem 1rem 3rem; }
        .hero { padding:2rem 1.35rem; min-height:230px; }
        .metric-grid, .flow { grid-template-columns:repeat(2,1fr); }
    }
    @media (max-width: 560px) {
        .hero h1 { font-size:2.35rem; }
        .metric-grid { grid-template-columns:1fr 1fr; }
        .metric-card { min-height:100px; }
        .flow { grid-template-columns:1fr; }
        .topline-title { display:none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading crop intelligence model...")
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


def render_topline(section: str) -> None:
    st.markdown(
        f"""
        <div class="topline">
            <div class="topline-title">KshetraSense / {html.escape(section)}</div>
            <div class="live-pill"><span class="status-dot"></span> MODEL ONLINE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(kicker: str, title: str, accent: str, copy: str, tags: list[str]) -> None:
    tag_markup = "".join(f'<span class="hero-tag">{html.escape(tag)}</span>' for tag in tags)
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-content">
                <div class="eyebrow">{html.escape(kicker)}</div>
                <h1>{html.escape(title)} <span>{html.escape(accent)}</span></h1>
                <p>{html.escape(copy)}</p>
                <div class="hero-tags">{tag_markup}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def metric_cards(cards: list[tuple[str, str, str, str]]) -> None:
    markup = "".join(
        f'<div class="metric-card" style="--tone:{tone}">'
        f'<div class="metric-kicker">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-foot">{html.escape(foot)}</div>'
        '</div>'
        for label, value, foot, tone in cards
    )
    st.markdown(f'<div class="metric-grid">{markup}</div>', unsafe_allow_html=True)


def section_heading(title: str, copy: str) -> None:
    st.markdown(
        f'<div class="section-head"><h2>{html.escape(title)}</h2><p>{html.escape(copy)}</p></div>',
        unsafe_allow_html=True,
    )


def render_ranked_result(ranked: pd.DataFrame) -> None:
    best_crop = str(ranked.iloc[0]["Crop"])
    best_key = best_crop.lower().replace(" ", "")
    best_confidence = float(ranked.iloc[0]["Confidence"])
    rows = []
    for index, row in ranked.iterrows():
        confidence = float(row["Confidence"])
        rows.append(
            f'<div class="rank-row">'
            f'<div class="rank-no">0{index + 1}</div>'
            '<div>'
            f'<div class="rank-name">{html.escape(str(row["Crop"]))}</div>'
            f'<div class="rank-track"><div class="rank-fill" style="width:{confidence * 100:.1f}%"></div></div>'
            '</div>'
            f'<div class="rank-pct">{confidence:.1%}</div>'
            '</div>'
        )
    st.markdown(
        f"""
        <div class="result-shell">
            <div class="crop-token">{CROP_ICONS.get(best_key, 'KS')}</div>
            <div class="best-label">Recommended crop</div>
            <div class="crop-name">{html.escape(best_crop)}</div>
            <div class="confidence"><span class="status-dot"></span>{best_confidence:.1%} confidence</div>
        </div>
        <div class="rank-list">{''.join(rows)}</div>
        """,
        unsafe_allow_html=True,
    )


try:
    scaler, encoder, model, eda, comparison, importance = load_artifacts()
except Exception as exc:
    st.error("KshetraSense could not load its trained model artifacts.")
    st.exception(exc)
    st.stop()


best_model = comparison["models"][0]

with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-row">
                <div class="brand-mark">KS</div>
                <div><div class="brand-name">KshetraSense</div><div class="brand-sub">Crop intelligence system</div></div>
            </div>
        </div>
        <div class="nav-label">Workspace</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        ["Recommend", "Overview", "Model performance", "Data explorer"],
        label_visibility="collapsed",
    )
    st.markdown(
        """
        <div class="sidebar-status">
            <div class="status-row"><span class="status-dot"></span> Random Forest ready</div>
            <div class="sidebar-note">Trained on 2,200 field records across 22 crop classes and seven soil-climate signals.</div>
        </div>
        <div class="sidebar-note" style="padding:.2rem .55rem">Decision support only. Validate recommendations with regional agronomic guidance.</div>
        """,
        unsafe_allow_html=True,
    )


if page == "Recommend":
    render_topline("Recommendation engine")
    render_hero(
        "Precision agriculture / ML-01",
        "Read the field.",
        "Choose with confidence.",
        "Translate seven soil and climate measurements into a ranked crop shortlist, powered by a tuned Random Forest model.",
        ["7 input signals", "22 crop classes", "Top-5 ranking", "Instant inference"],
    )
    metric_cards(
        [
            ("Training records", f"{eda['shape'][0]:,}", "balanced crop dataset", "#7ee787"),
            ("Crop classes", str(len(eda["class_distribution"])), "recommendation targets", "#c8f169"),
            ("Input signals", str(len(FEATURES)), "soil + climate factors", "#f0c75e"),
            ("Test accuracy", f"{best_model['accuracy']:.2%}", "held-out evaluation", "#8bbdff"),
        ]
    )

    left, right = st.columns([1.08, .92], gap="large")
    with left:
        st.markdown('<div class="panel-title">Field profile</div><div class="panel-sub">Enter the latest laboratory and environmental measurements.</div>', unsafe_allow_html=True)
        with st.form("prediction_form"):
            st.markdown('<div class="mono-label">Soil nutrients / mg per kg</div>', unsafe_allow_html=True)
            nutrient_cols = st.columns(3)
            with nutrient_cols[0]:
                nitrogen = st.number_input("Nitrogen (N)", 0.0, 200.0, 90.0, 1.0)
            with nutrient_cols[1]:
                phosphorus = st.number_input("Phosphorus (P)", 0.0, 200.0, 42.0, 1.0)
            with nutrient_cols[2]:
                potassium = st.number_input("Potassium (K)", 0.0, 250.0, 43.0, 1.0)

            st.markdown('<div class="mono-label" style="margin-top:.9rem">Climate and chemistry</div>', unsafe_allow_html=True)
            climate_cols = st.columns(2)
            with climate_cols[0]:
                temperature = st.number_input("Temperature (deg C)", -10.0, 60.0, 21.0, .5)
                rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 203.0, 1.0)
            with climate_cols[1]:
                humidity = st.number_input("Humidity (%)", 0.0, 100.0, 82.0, 1.0)
                soil_ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, .1)
            submitted = st.form_submit_button("Run crop analysis", type="primary", use_container_width=True)

        if submitted:
            values = [nitrogen, phosphorus, potassium, temperature, humidity, soil_ph, rainfall]
            st.session_state["ranked_crops"] = predict_crops(values, scaler, encoder, model)

    with right:
        st.markdown('<div class="panel-title">Model output</div><div class="panel-sub">Ranked by probability across all available crop classes.</div>', unsafe_allow_html=True)
        if "ranked_crops" in st.session_state:
            render_ranked_result(st.session_state["ranked_crops"])
            st.markdown('<div class="notice" style="margin-top:1rem">Confidence expresses the model\'s relative certainty for these inputs. It does not estimate yield, market value, or local disease risk.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="empty-result">
                    <div class="empty-icon">AI</div>
                    <div class="empty-title">Awaiting field data</div>
                    <div class="empty-copy">Complete the profile and run the analysis to reveal the strongest crop matches.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

elif page == "Overview":
    render_topline("System overview")
    render_hero(
        "Project intelligence / DS-01",
        "From field signals to",
        "clear decisions.",
        "KshetraSense combines a balanced crop dataset, reproducible preprocessing, and probabilistic classification in one interpretable workflow.",
        ["Supervised learning", "Balanced dataset", "Reproducible pipeline", "Decision support"],
    )
    metric_cards(
        [
            ("Observations", f"{eda['shape'][0]:,}", "training records", "#7ee787"),
            ("Target classes", str(len(eda["class_distribution"])), "supported crops", "#c8f169"),
            ("Champion model", best_model["model"], "selected by evaluation", "#f0c75e"),
            ("Test accuracy", f"{best_model['accuracy']:.2%}", "20% holdout set", "#8bbdff"),
        ]
    )
    section_heading("Inference pipeline", "Four controlled stages turn raw measurements into ranked recommendations.")
    st.markdown(
        """
        <div class="flow">
            <div class="flow-step"><div class="flow-num">01 / INPUT</div><div class="flow-title">Field measurements</div><div class="flow-copy">Capture N, P, K, temperature, humidity, pH, and rainfall.</div></div>
            <div class="flow-step"><div class="flow-num">02 / TRANSFORM</div><div class="flow-title">Standardization</div><div class="flow-copy">Apply the fitted scaler used during model training.</div></div>
            <div class="flow-step"><div class="flow-num">03 / INFER</div><div class="flow-title">Random Forest</div><div class="flow-copy">Estimate probability across all 22 crop classes.</div></div>
            <div class="flow-step"><div class="flow-num">04 / RANK</div><div class="flow-title">Decision shortlist</div><div class="flow-copy">Return the five most compatible crops in descending order.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    section_heading("Class coverage", "Each crop contributes the same number of samples to reduce class imbalance.")
    distribution = pd.Series(eda["class_distribution"], name="Samples").sort_index()
    st.bar_chart(distribution, color="#7ee787", height=360)
    st.markdown('<div class="notice">Real-world performance can shift with geography, season, measurement quality, and farming practices not represented in the dataset.</div>', unsafe_allow_html=True)

elif page == "Model performance":
    render_topline("Model performance")
    render_hero(
        "Evaluation lab / ML-02",
        "Evidence before",
        "recommendation.",
        "Compare candidate classifiers on a shared holdout set and inspect what drives the selected Random Forest model.",
        ["4 candidate models", "Shared test split", "Cross-validation", "Feature importance"],
    )
    models = pd.DataFrame(comparison["models"])
    metric_cards(
        [
            ("Best model", best_model["model"], "highest holdout score", "#7ee787"),
            ("Accuracy", f"{best_model['accuracy']:.2%}", "correct predictions", "#c8f169"),
            ("F1 score", f"{best_model['f1_score']:.2%}", "precision / recall balance", "#f0c75e"),
            ("CV mean", f"{best_model['cv_mean']:.2%}", "cross-validation result", "#8bbdff"),
        ]
    )
    section_heading("Candidate comparison", "Every classifier was evaluated against the same held-out observations.")
    display = models[["model", "accuracy", "precision", "recall", "f1_score", "cv_mean", "cv_std"]].copy()
    display.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 score", "CV mean", "CV std"]
    st.dataframe(
        display.style.format({column: "{:.2%}" for column in display.columns if column != "Model"}),
        hide_index=True,
        use_container_width=True,
    )
    chart_col, importance_col = st.columns(2, gap="large")
    with chart_col:
        section_heading("Holdout accuracy", "Direct comparison of correct classification rates.")
        st.bar_chart(display.set_index("Model")["Accuracy"], horizontal=True, color="#7ee787", height=330)
    with importance_col:
        section_heading("Feature reliance", "Relative importance within the Random Forest.")
        feature_importance = pd.Series(importance["Random Forest"], name="Importance").sort_values()
        st.bar_chart(feature_importance, horizontal=True, color="#f0c75e", height=330)
    st.markdown('<div class="notice">Feature importance measures predictive reliance, not a causal effect on crop growth or yield.</div>', unsafe_allow_html=True)

else:
    render_topline("Data explorer")
    render_hero(
        "Dataset observatory / DS-02",
        "See the signals",
        "behind the model.",
        "Explore distributions, descriptive statistics, and relationships across every soil and climate feature used for inference.",
        ["7 dimensions", "20-bin histograms", "Summary statistics", "Correlation matrix"],
    )
    metric_cards(
        [
            ("Rows", f"{eda['shape'][0]:,}", "complete observations", "#7ee787"),
            ("Features", str(len(FEATURES)), "model inputs", "#c8f169"),
            ("Missing values", "0", "clean training matrix", "#f0c75e"),
            ("Classes", str(len(eda["class_distribution"])), "balanced labels", "#8bbdff"),
        ]
    )
    section_heading("Feature distribution", "Select a signal to inspect its frequency across the training dataset.")
    feature = st.selectbox("Feature", FEATURES, format_func=lambda value: FEATURE_LABELS[value])
    histogram = eda["histograms"][feature]
    edges = histogram["bin_edges"]
    histogram_frame = pd.DataFrame(
        {
            "Range midpoint": [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)],
            "Samples": histogram["counts"],
        }
    ).set_index("Range midpoint")
    st.bar_chart(histogram_frame, color="#7ee787", height=360)

    stat_col, corr_col = st.columns([.84, 1.16], gap="large")
    with stat_col:
        section_heading("Feature summary", "Central tendency and spread for each signal.")
        summary = pd.DataFrame(eda["statistics"]).T
        summary.index.name = "Feature"
        st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)
    with corr_col:
        section_heading("Correlation matrix", "Pairwise linear relationships between inputs.")
        correlation = eda["correlation_matrix"]
        correlation_frame = pd.DataFrame(
            correlation["values"], index=correlation["columns"], columns=correlation["columns"]
        )
        st.dataframe(
            correlation_frame.round(2),
            use_container_width=True,
        )
