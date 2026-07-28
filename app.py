"""KshetraSense: an interactive crop recommendation dashboard."""

from __future__ import annotations

import html
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ml_core import (
    assess_ood,
    crop_fit,
    load_model_bundle,
    local_sensitivity,
    predict_ranked,
    validate_external_dataframe,
)


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATASET_PATH = ROOT / "dataset" / "Crop_recommendation.csv"
NOTEBOOK_PATH = ROOT / "notebooks" / "kshetra_sense.ipynb"
TRAINING_SCRIPT_PATH = ROOT / "training" / "train_models.py"
CORE_PATH = ROOT / "ml_core.py"
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
CHART_COLORS = [
    "#7ee787", "#c8f169", "#f0c75e", "#8bbdff", "#bc8cff", "#ff7b72",
    "#ffa657", "#56d4dd", "#d2a8ff", "#79c0ff", "#a5d6ff", "#f2cc60",
    "#aff5b4", "#db61a2", "#ff9bce", "#39c5cf", "#d29922", "#58a6ff",
    "#3fb950", "#e3b341", "#f85149", "#8957e5",
]


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
    [data-testid="stToolbar"] {
        visibility: visible;
        background: transparent;
    }
    footer { display: none; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a110c 0%, #0c140f 55%, #080d0a 100%);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 1.55rem 1rem 1.25rem;
    }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"] {
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999 !important;
    }
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapsedControl"] button {
        width: 34px !important; height: 34px !important;
        border-radius: 10px !important;
        color: var(--green) !important;
        background: #132019 !important;
        border: 1px solid rgba(126,231,135,.25) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,.3), 0 0 18px rgba(126,231,135,.08) !important;
        transition: transform .18s ease, background .18s ease !important;
    }
    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover {
        transform: scale(1.06);
        color: var(--lime) !important;
        background: #1a2b20 !important;
        border-color: rgba(200,241,105,.45) !important;
    }
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stSidebarCollapsedControl"] svg {
        fill: currentColor !important;
        width: 21px !important; height: 21px !important;
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

    /* Prediction console + widgets */
    div[data-testid="stForm"] {
        position:relative; overflow:hidden;
        border:1px solid #293a2e !important;
        border-radius:20px !important;
        padding:1.2rem 1.25rem 1.25rem !important;
        background:
            radial-gradient(circle at 100% 0%,rgba(126,231,135,.065),transparent 16rem),
            linear-gradient(145deg,#101813,#111c15) !important;
        box-shadow:inset 0 1px 0 rgba(255,255,255,.035),0 18px 50px rgba(0,0,0,.18);
    }
    .form-console-head {
        display:flex; align-items:flex-start; justify-content:space-between; gap:1rem;
        padding-bottom:1rem; margin-bottom:1.05rem; border-bottom:1px solid var(--line-soft);
    }
    .form-console-kicker { color:var(--green); font:600 .61rem 'JetBrains Mono'; letter-spacing:.12em; text-transform:uppercase; }
    .form-console-title { font:700 1.08rem 'Space Grotesk'; letter-spacing:-.025em; margin-top:.24rem; }
    .form-console-copy { color:var(--muted); font-size:.7rem; margin-top:.22rem; }
    .form-ready {
        flex:0 0 auto; display:flex; align-items:center; gap:.38rem; padding:.36rem .58rem;
        border:1px solid rgba(126,231,135,.18); border-radius:999px;
        background:rgba(126,231,135,.065); color:#aed9b3;
        font:600 .6rem 'JetBrains Mono'; text-transform:uppercase; letter-spacing:.06em;
    }
    .field-group-head {
        display:flex; align-items:center; gap:.62rem; margin:.1rem 0 .5rem;
        padding:.58rem .66rem; border-radius:10px;
        border:1px solid rgba(255,255,255,.045); background:rgba(255,255,255,.018);
    }
    .field-group-icon {
        width:31px; height:31px; border-radius:9px; display:grid; place-items:center;
        color:var(--green); border:1px solid rgba(126,231,135,.18);
        background:rgba(126,231,135,.07); font:700 .58rem 'JetBrains Mono';
    }
    .field-group-title { color:#dce8df; font:700 .76rem 'Space Grotesk'; }
    .field-group-copy { color:#6f8376; font-size:.61rem; margin-top:.12rem; }
    .form-foot {
        display:flex; align-items:center; gap:.48rem; margin:.35rem 0 .75rem;
        color:#6f8376; font-size:.65rem; line-height:1.45;
    }
    .form-foot::before { content:'i'; width:17px; height:17px; display:grid; place-items:center; flex:0 0 auto; border-radius:50%; border:1px solid #405646; color:#91a598; font:600 .57rem 'JetBrains Mono'; }
    [data-testid="stSlider"] { padding:.25rem .08rem .48rem; }
    [data-testid="stSlider"] label p {
        color:#b9c8bd !important; font-size:.72rem !important; font-weight:700 !important;
        letter-spacing:.01em;
    }
    [data-testid="stSlider"] [role="slider"] {
        width:17px !important; height:17px !important;
        background:var(--lime) !important; border:3px solid #122017 !important;
        box-shadow:0 0 0 2px rgba(200,241,105,.45),0 0 16px rgba(126,231,135,.3) !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        height:5px !important; border-radius:999px !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div:first-child {
        background:linear-gradient(90deg,#65d77a,#c8f169) !important;
    }
    [data-testid="stSlider"] [data-testid="stTickBar"] {
        color:#607267 !important; font:500 .58rem 'JetBrains Mono' !important;
    }
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
        min-height:48px; border-radius:12px; border:1px solid #98dd78;
        background:linear-gradient(135deg,#b6ed7a,#79d783); color:#0a150c;
        font-family:'Space Grotesk'; font-weight:800; letter-spacing:-.01em;
        box-shadow:0 8px 24px rgba(126,231,135,.12); transition:.18s ease;
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
    .notebook-intro {
        display:flex; align-items:center; gap:.85rem; padding:1rem 1.1rem;
        border:1px solid var(--line); border-radius:14px;
        background:linear-gradient(145deg,#111a14,#151f18); margin:1rem 0 1.3rem;
    }
    .notebook-logo {
        width:42px; height:42px; border-radius:12px; display:grid; place-items:center;
        color:#0b130d; background:linear-gradient(135deg,#f0c75e,#ffa657);
        font:800 .72rem 'JetBrains Mono'; box-shadow:0 8px 24px rgba(240,199,94,.14);
    }
    .notebook-title { font:700 .93rem 'Space Grotesk'; }
    .notebook-copy { color:var(--muted); font-size:.72rem; margin-top:.18rem; }
    .cell-label {
        display:flex; align-items:center; justify-content:space-between;
        margin:1.15rem 0 .38rem; color:#708378;
        font:600 .62rem 'JetBrains Mono'; letter-spacing:.1em; text-transform:uppercase;
    }
    .cell-kind { color:var(--green); }
    [data-testid="stCodeBlock"] {
        border:1px solid var(--line) !important; border-radius:12px !important;
        box-shadow:inset 3px 0 0 rgba(126,231,135,.36);
        overflow:hidden;
    }
    [data-baseweb="tab-list"] {
        gap:.35rem; padding:.35rem; border:1px solid var(--line); border-radius:12px;
        background:var(--surface);
    }
    [data-baseweb="tab"] { border-radius:8px; color:var(--muted); font-weight:700; }
    [data-baseweb="tab"][aria-selected="true"] { color:var(--green); background:rgba(126,231,135,.08); }

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


@st.cache_resource(show_spinner="Verifying crop intelligence model...")
def load_artifacts():
    """Load the combined pipeline only after checksum and schema validation."""
    bundle = load_model_bundle(MODELS_DIR)
    documents = {}
    for name in ("eda", "model_comparison", "feature_importance"):
        documents[name] = json.loads((MODELS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return bundle, documents["eda"], documents["model_comparison"], documents["feature_importance"]


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load and validate the compact dataset used by the EDA workspace."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET_PATH.name}")
    frame = pd.read_csv(DATASET_PATH)
    expected = [*FEATURES, "label"]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")
    frame["Crop"] = frame["label"].str.replace("beans", " beans").str.title()
    return frame


def style_chart(chart: alt.Chart, height: int = 340) -> alt.Chart:
    """Apply the dashboard's dark visual language to an Altair chart."""
    return (
        chart.properties(height=height)
        .configure_view(strokeOpacity=0)
        .configure_axis(
            gridColor="#223027",
            gridOpacity=.7,
            domainColor="#34483a",
            tickColor="#34483a",
            labelColor="#8fa398",
            titleColor="#c5d2c8",
            labelFont="DM Sans",
            titleFont="DM Sans",
        )
        .configure_legend(
            labelColor="#9caf9f",
            titleColor="#d5e1d8",
            labelFont="DM Sans",
            titleFont="DM Sans",
            orient="bottom",
        )
    )


def predict_crops(values: list[float], bundle: object) -> pd.DataFrame:
    """Return the five most likely crops using calibrated probabilities."""
    ranked = predict_ranked(bundle, values, top_k=5)
    return pd.DataFrame({
        "Crop": [row["crop"].replace("beans", " beans").title() for row in ranked],
        "Probability": [row["probability"] for row in ranked],
    })


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


def render_notebook_cells(notebook: dict) -> None:
    """Render notebook markdown and code cells as a Jupyter-style document."""
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            st.markdown(source)
            continue
        if cell.get("cell_type") == "code":
            st.markdown(
                f'<div class="cell-label"><span>In [{index}]</span><span class="cell-kind">Python</span></div>',
                unsafe_allow_html=True,
            )
            st.code(source, language="python", line_numbers=True)


def render_ranked_result(ranked: pd.DataFrame) -> None:
    best_crop = str(ranked.iloc[0]["Crop"])
    best_key = best_crop.lower().replace(" ", "")
    best_confidence = float(ranked.iloc[0]["Probability"])
    rows = []
    for index, row in ranked.iterrows():
        confidence = float(row["Probability"])
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
            <div class="confidence"><span class="status-dot"></span>{best_confidence:.1%} calibrated probability</div>
        </div>
        <div class="rank-list">{''.join(rows)}</div>
        """,
        unsafe_allow_html=True,
    )


try:
    bundle, eda, comparison, importance = load_artifacts()
except Exception as exc:
    st.error("KshetraSense could not load its trained model artifacts.")
    st.exception(exc)
    st.stop()


champion = comparison["champion"]
test_metrics = champion["test_metrics"]
best_model = next(row for row in comparison["models"] if row["model"] == champion["model"])

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
        ["Recommend", "Overview", "Model performance", "Data explorer", "Jupyter notebook"],
        label_visibility="collapsed",
    )
    st.markdown(
        """
        <div class="sidebar-status">
            <div class="status-row"><span class="status-dot"></span> Verified model ready</div>
            <div class="sidebar-note">Leakage-free pipeline, calibrated probabilities, input safeguards, and seven soil-climate signals.</div>
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
        "Translate seven soil and climate measurements into a guarded crop shortlist with calibrated probabilities and novelty checks.",
        ["7 input signals", "22 crop classes", "Calibrated ranking", "OOD guardrail"],
    )
    metric_cards(
        [
            ("Training records", f"{eda['shape'][0]:,}", "balanced crop dataset", "#7ee787"),
            ("Crop classes", str(len(eda["class_distribution"])), "recommendation targets", "#c8f169"),
            ("Input signals", str(len(FEATURES)), "soil + climate factors", "#f0c75e"),
            ("Test accuracy", f"{test_metrics['accuracy']:.2%}", "untouched holdout", "#8bbdff"),
        ]
    )

    left, right = st.columns([1.12, .88], gap="large")
    with left:
        with st.form("prediction_form"):
            st.markdown(
                """
                <div class="form-console-head">
                    <div><div class="form-console-kicker">Field signal console</div>
                    <div class="form-console-title">Build your field profile</div>
                    <div class="form-console-copy">Tune each signal using the latest available measurements.</div></div>
                    <div class="form-ready"><span class="status-dot"></span> 7 signals</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            soil_col, climate_col = st.columns(2, gap="large")
            with soil_col:
                st.markdown(
                    '<div class="field-group-head"><div class="field-group-icon">NPK</div><div><div class="field-group-title">Soil profile</div><div class="field-group-copy">Nutrients and acidity</div></div></div>',
                    unsafe_allow_html=True,
                )
                nitrogen = st.slider("Nitrogen (N)", int(bundle.ood_profile["ranges"]["N"]["min"]), int(bundle.ood_profile["ranges"]["N"]["max"]), 90, 1, format="%d mg/kg")
                phosphorus = st.slider("Phosphorus (P)", int(bundle.ood_profile["ranges"]["P"]["min"]), int(bundle.ood_profile["ranges"]["P"]["max"]), 42, 1, format="%d mg/kg")
                potassium = st.slider("Potassium (K)", int(bundle.ood_profile["ranges"]["K"]["min"]), int(bundle.ood_profile["ranges"]["K"]["max"]), 43, 1, format="%d mg/kg")
                soil_ph = st.slider("Soil pH", float(bundle.ood_profile["ranges"]["ph"]["min"]), float(bundle.ood_profile["ranges"]["ph"]["max"]), 6.5, .1, format="%.1f pH")
            with climate_col:
                st.markdown(
                    '<div class="field-group-head"><div class="field-group-icon">ENV</div><div><div class="field-group-title">Climate profile</div><div class="field-group-copy">Weather and water</div></div></div>',
                    unsafe_allow_html=True,
                )
                temperature = st.slider("Temperature", float(bundle.ood_profile["ranges"]["temperature"]["min"]), float(bundle.ood_profile["ranges"]["temperature"]["max"]), 21.0, .5, format="%.1f deg C")
                humidity = st.slider("Humidity", int(bundle.ood_profile["ranges"]["humidity"]["min"]), int(bundle.ood_profile["ranges"]["humidity"]["max"]), 82, 1, format="%d%%")
                rainfall = st.slider("Rainfall", int(bundle.ood_profile["ranges"]["rainfall"]["min"]), int(bundle.ood_profile["ranges"]["rainfall"]["max"]), 203, 1, format="%d mm")
            st.markdown(
                '<div class="form-foot">Use laboratory soil values where possible. Sliders support arrow keys for precise adjustment.</div>',
                unsafe_allow_html=True,
            )
            submitted = st.form_submit_button("Analyze field and recommend crops", type="primary", width="stretch")

        if submitted:
            values = [nitrogen, phosphorus, potassium, temperature, humidity, soil_ph, rainfall]
            assessment = assess_ood(values, bundle.ood_profile)
            st.session_state["last_values"] = values
            st.session_state["last_ood"] = assessment
            events = st.session_state.setdefault("prediction_events", [])
            event = {"ood_status": assessment["status"], "distance": assessment["distance"]}
            if assessment["status"] == "abstain":
                st.session_state.pop("ranked_crops", None)
                st.session_state.pop("local_sensitivity", None)
                event["outcome"] = "abstained_ood"
            else:
                ranked = predict_crops(values, bundle)
                event["top_probability"] = float(ranked.iloc[0]["Probability"])
                if event["top_probability"] < bundle.manifest["confidence_threshold"]:
                    st.session_state.pop("ranked_crops", None)
                    st.session_state.pop("local_sensitivity", None)
                    event["outcome"] = "abstained_low_probability"
                else:
                    st.session_state["ranked_crops"] = ranked
                    st.session_state["local_sensitivity"] = local_sensitivity(bundle, values)
                    event["outcome"] = "recommended"
            events.append(event)

    with right:
        st.markdown('<div class="panel-title">Model output</div><div class="panel-sub">Ranked by probability across all available crop classes.</div>', unsafe_allow_html=True)
        assessment = st.session_state.get("last_ood")
        if assessment and assessment["status"] == "abstain":
            features = assessment["outside_features"] or assessment["unusual_features"]
            st.error("No recommendation returned: this field profile is outside the model's supported data region.")
            if features:
                st.caption("Review these measurements: " + ", ".join(FEATURE_LABELS[item] for item in features))
        elif assessment and "ranked_crops" not in st.session_state:
            st.warning(f"No recommendation returned: the highest calibrated probability was below {bundle.manifest['confidence_threshold']:.0%}.")
        elif "ranked_crops" in st.session_state:
            render_ranked_result(st.session_state["ranked_crops"])
            if assessment and assessment["status"] == "warning":
                st.warning("This is an unusual combination relative to training data. Treat the ranking with extra caution.")
            with st.expander("Why this recommendation?", expanded=False):
                st.caption("One-at-a-time sensitivity shows association, not causation. Positive values indicate signals that most support the top-ranked crop versus the training mean.")
                sensitivity = st.session_state.get("local_sensitivity", pd.DataFrame())
                if not sensitivity.empty:
                    chart_data = sensitivity.assign(Label=sensitivity["feature"].map(FEATURE_LABELS)).set_index("Label")
                    st.bar_chart(chart_data["probability_change"], horizontal=True, color="#f0c75e")
                    best_crop_key = str(st.session_state["ranked_crops"].iloc[0]["Crop"]).lower().replace(" ", "")
                    fit = crop_fit(bundle, best_crop_key, st.session_state["last_values"])
                    st.dataframe(fit.rename(columns={"feature": "Signal", "value": "Input", "crop_average": "Crop average", "standard_deviations": "Std. deviations"}), hide_index=True, width="stretch")
            st.markdown('<div class="notice" style="margin-top:1rem">Calibrated probability expresses model uncertainty. It does not estimate yield, profit, disease risk, or local suitability.</div>', unsafe_allow_html=True)
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
        "KshetraSense combines a balanced crop dataset, leakage-free model selection, calibrated classification, and guarded inference in one reproducible workflow.",
        ["Training-only CV", "Calibrated model", "OOD detection", "Decision support"],
    )
    metric_cards(
        [
            ("Observations", f"{eda['shape'][0]:,}", "training records", "#7ee787"),
            ("Target classes", str(len(eda["class_distribution"])), "supported crops", "#c8f169"),
            ("Champion model", champion["model"], "selected by CV macro F1", "#f0c75e"),
            ("Test accuracy", f"{test_metrics['accuracy']:.2%}", "untouched 20% holdout", "#8bbdff"),
        ]
    )
    section_heading("Inference pipeline", "Four controlled stages turn raw measurements into ranked recommendations.")
    st.markdown(
        """
        <div class="flow">
            <div class="flow-step"><div class="flow-num">01 / INPUT</div><div class="flow-title">Field measurements</div><div class="flow-copy">Capture N, P, K, temperature, humidity, pH, and rainfall.</div></div>
            <div class="flow-step"><div class="flow-num">02 / GUARD</div><div class="flow-title">Novelty detection</div><div class="flow-copy">Reject values outside observed limits and flag unusual multivariate profiles.</div></div>
            <div class="flow-step"><div class="flow-num">03 / INFER</div><div class="flow-title">Calibrated pipeline</div><div class="flow-copy">Estimate calibrated probability across all 22 crop classes.</div></div>
            <div class="flow-step"><div class="flow-num">04 / ABSTAIN</div><div class="flow-title">Safe shortlist</div><div class="flow-copy">Return a ranking only when novelty and probability thresholds are satisfied.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    section_heading("Class coverage", "Each crop contributes the same number of samples to reduce class imbalance.")
    distribution = pd.Series(eda["class_distribution"], name="Samples").sort_index()
    st.bar_chart(distribution, color="#7ee787", height=360)
    st.markdown('<div class="notice"><strong>Validation scope:</strong> internal stratified holdout only. The dataset has no geography, season, farm, yield, cost, or prospective outcome fields, so this is not evidence of field-level effectiveness. Real-world performance can shift with region, season, measurement quality, and farming practice.</div>', unsafe_allow_html=True)

elif page == "Model performance":
    render_topline("Model performance")
    render_hero(
        "Evaluation lab / ML-02",
        "Evidence before",
        "recommendation.",
        "Inspect training-only model selection, untouched test performance, probability calibration, class-level errors, and external validation.",
        ["CV-only selection", "Untouched holdout", "Calibration", "External validation"],
    )
    models = pd.DataFrame(comparison["models"])
    metric_cards(
        [
            ("Champion", champion["model"], "chosen before test evaluation", "#7ee787"),
            ("Macro F1", f"{test_metrics['macro_f1']:.2%}", f"95% CI {test_metrics['macro_f1_ci_95'][0]:.2%}–{test_metrics['macro_f1_ci_95'][1]:.2%}", "#c8f169"),
            ("Balanced accuracy", f"{test_metrics['balanced_accuracy']:.2%}", "equal weight per class", "#f0c75e"),
            ("Top-3 accuracy", f"{test_metrics['top_3_accuracy']:.2%}", "correct crop in shortlist", "#8bbdff"),
        ]
    )
    st.markdown(f'<div class="notice"><strong>Selection policy:</strong> {html.escape(comparison["selection_policy"])} The holdout contains {comparison["test_records"]} records and was evaluated once after calibration.</div>', unsafe_allow_html=True)

    section_heading("Candidate comparison", "Only cross-validation on the 1,760-record training partition was used to choose the champion.")
    display = models[["model", "cv_macro_f1", "cv_macro_f1_std", "cv_balanced_accuracy", "cv_accuracy"]].copy()
    display.columns = ["Model", "CV macro F1", "CV F1 std", "CV balanced accuracy", "CV accuracy"]
    st.dataframe(
        display.style.format({column: "{:.2%}" for column in display.columns if column != "Model"}),
        hide_index=True,
        width="stretch",
    )
    chart_col, importance_col = st.columns(2, gap="large")
    with chart_col:
        section_heading("Cross-validated macro F1", "A dummy baseline makes the improvement visible in context.")
        st.bar_chart(display.set_index("Model")["CV macro F1"], horizontal=True, color="#7ee787", height=330)
    with importance_col:
        section_heading("Permutation importance", "Test-set macro F1 decrease after each signal is shuffled.")
        feature_importance = pd.DataFrame(importance["rows"]).assign(
            Signal=lambda frame: frame["feature"].map(FEATURE_LABELS)
        ).set_index("Signal")["mean"].sort_values()
        st.bar_chart(feature_importance, horizontal=True, color="#f0c75e", height=330)
    st.markdown('<div class="notice">Permutation importance and local sensitivity measure predictive association—not causal effects on crop growth or yield.</div>', unsafe_allow_html=True)

    section_heading("Probability quality", "Accuracy alone cannot show whether a model's probabilities deserve trust.")
    metric_cards([
        ("Log loss", f"{test_metrics['log_loss']:.4f}", "lower is better", "#7ee787"),
        ("Multiclass Brier", f"{test_metrics['multiclass_brier']:.4f}", "probability error", "#c8f169"),
        ("Calibration error", f"{test_metrics['expected_calibration_error']:.2%}", "confidence gap", "#f0c75e"),
        ("Accuracy", f"{test_metrics['accuracy']:.2%}", f"95% CI {test_metrics['accuracy_ci_95'][0]:.2%}–{test_metrics['accuracy_ci_95'][1]:.2%}", "#8bbdff"),
    ])
    reliability = pd.DataFrame(champion["reliability"]).dropna(subset=["mean_confidence", "accuracy"])
    if not reliability.empty:
        reliability_chart = alt.Chart(reliability).mark_line(point=alt.OverlayMarkDef(size=90), color="#c8f169", strokeWidth=3).encode(
            x=alt.X("mean_confidence:Q", title="Mean calibrated probability", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("accuracy:Q", title="Observed accuracy", scale=alt.Scale(domain=[0, 1])),
            size=alt.Size("count:Q", title="Records"),
            tooltip=[alt.Tooltip("count:Q", title="Records"), alt.Tooltip("mean_confidence:Q", format=".2%"), alt.Tooltip("accuracy:Q", format=".2%")],
        )
        ideal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="#607568", strokeDash=[6, 5]).encode(x="x:Q", y="y:Q")
        st.altair_chart(style_chart(ideal + reliability_chart, 330), width="stretch")

    confusion_col, classes_col = st.columns([1.15, .85], gap="large")
    with confusion_col:
        section_heading("Labelled confusion matrix", "Only two of 440 internal holdout records were misclassified.")
        labels = comparison["label_classes"]
        confusion_rows = [{"Actual": labels[row], "Predicted": labels[column], "Count": value}
                          for row, values in enumerate(champion["confusion_matrix"])
                          for column, value in enumerate(values)]
        confusion_chart = alt.Chart(pd.DataFrame(confusion_rows)).mark_rect(cornerRadius=2).encode(
            x=alt.X("Predicted:N", sort=labels, axis=alt.Axis(labelAngle=-55, labelLimit=90)),
            y=alt.Y("Actual:N", sort=labels),
            color=alt.Color("Count:Q", scale=alt.Scale(range=["#142019", "#7ee787"])),
            tooltip=["Actual:N", "Predicted:N", "Count:Q"],
        )
        st.altair_chart(style_chart(confusion_chart, 520), width="stretch")
    with classes_col:
        section_heading("Per-crop performance", "Macro metrics prevent large classes from hiding weak crops.")
        class_table = pd.DataFrame(champion["per_class"]).rename(columns={
            "crop": "Crop", "precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Support"
        })
        st.dataframe(class_table.style.format({"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}", "Support": "{:.0f}"}),
                     hide_index=True, width="stretch", height=520)

    section_heading("External validation", "Upload a labelled CSV from a different region, season, or collection process to test transportability without retraining.")
    uploaded = st.file_uploader("Labelled CSV", type=["csv"], help="Required columns: N, P, K, temperature, humidity, ph, rainfall, label")
    if uploaded is not None:
        if uploaded.size > 5 * 1024 * 1024:
            st.error("Please upload a CSV smaller than 5 MB.")
        else:
            try:
                external = pd.read_csv(uploaded)
                external_result = validate_external_dataframe(bundle, external)
                metric_cards([
                    ("Records", f"{external_result['records']:,}", "external observations", "#7ee787"),
                    ("Macro F1", f"{external_result['macro_f1']:.2%}", "external labels", "#c8f169"),
                    ("Top-3 accuracy", f"{external_result['top_3_accuracy']:.2%}", "external shortlist", "#f0c75e"),
                    ("OOD rate", f"{external_result['ood_warning_rate']:.2%}", "novel field profiles", "#8bbdff"),
                ])
                result_frame = external.copy()
                result_frame["model_prediction"] = external_result["predictions"]
                st.download_button("Download validation predictions", result_frame.to_csv(index=False),
                                   "kshetra_external_validation.csv", "text/csv", width="stretch")
            except Exception as exc:
                st.error(f"Could not validate this CSV: {exc}")

    with st.expander("Artifact and session diagnostics"):
        st.json({
            "artifact_schema": bundle.manifest["schema_version"],
            "trained_utc": bundle.manifest["created_utc"],
            "dataset_sha256": bundle.manifest["dataset_sha256"],
            "model_sha256": bundle.manifest["model_sha256"],
            "versions": bundle.manifest["versions"],
            "session_predictions": len(st.session_state.get("prediction_events", [])),
            "runtime_warnings": list(bundle.warnings),
        })

elif page == "Data explorer":
    dataset = load_dataset()
    render_topline("Data explorer")
    render_hero(
        "Dataset observatory / DS-02",
        "See the signals",
        "behind the model.",
        "Explore distributions, descriptive statistics, and relationships across every soil and climate feature used for inference.",
        ["7 dimensions", "Interactive distributions", "Crop comparisons", "Relationship maps"],
    )
    metric_cards(
        [
            ("Rows", f"{eda['shape'][0]:,}", "complete observations", "#7ee787"),
            ("Features", str(len(FEATURES)), "model inputs", "#c8f169"),
            ("Missing values", "0", "clean training matrix", "#f0c75e"),
            ("Classes", str(len(eda["class_distribution"])), "balanced labels", "#8bbdff"),
        ]
    )
    section_heading("Feature distribution", "Inspect the shape, spread, and crop-level variation of any model input.")
    feature = st.selectbox(
        "Feature to profile",
        FEATURES,
        format_func=lambda value: FEATURE_LABELS[value],
        key="distribution_feature",
    )
    feature_stats = eda["statistics"][feature]
    metric_cards(
        [
            ("Mean", f"{feature_stats['mean']:.2f}", FEATURE_LABELS[feature], "#7ee787"),
            ("Median", f"{feature_stats['median']:.2f}", "50th percentile", "#c8f169"),
            ("Std. deviation", f"{feature_stats['std']:.2f}", "dataset spread", "#f0c75e"),
            ("Observed range", f"{feature_stats['min']:.1f}–{feature_stats['max']:.1f}", "minimum to maximum", "#8bbdff"),
        ]
    )

    distribution_chart = (
        alt.Chart(dataset)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, opacity=.9)
        .encode(
            x=alt.X(
                f"{feature}:Q",
                bin=alt.Bin(maxbins=24),
                title=FEATURE_LABELS[feature],
            ),
            y=alt.Y("count():Q", title="Records"),
            color=alt.Color(
                "count():Q",
                title="Density",
                scale=alt.Scale(range=["#173622", "#7ee787", "#c8f169", "#f0c75e"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(f"{feature}:Q", bin=alt.Bin(maxbins=24), title=FEATURE_LABELS[feature]),
                alt.Tooltip("count():Q", title="Records"),
            ],
        )
    )
    st.altair_chart(style_chart(distribution_chart, 350), width="stretch")

    section_heading("Crop-wise spread", "Compare the full value range and median for every crop class.")
    box_chart = (
        alt.Chart(dataset)
        .mark_boxplot(size=13, extent="min-max", median={"color": "#edf5ef", "strokeWidth": 1.5})
        .encode(
            x=alt.X(f"{feature}:Q", title=FEATURE_LABELS[feature]),
            y=alt.Y("Crop:N", title=None, sort=alt.EncodingSortField(field=feature, op="median")),
            color=alt.Color(
                "Crop:N",
                scale=alt.Scale(range=CHART_COLORS),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Crop:N"),
                alt.Tooltip(f"min({feature}):Q", title="Minimum", format=".2f"),
                alt.Tooltip(f"median({feature}):Q", title="Median", format=".2f"),
                alt.Tooltip(f"max({feature}):Q", title="Maximum", format=".2f"),
            ],
        )
    )
    st.altair_chart(style_chart(box_chart, 540), width="stretch")

    section_heading("Relationship explorer", "Plot any two inputs and color the observations by crop.")
    relation_controls = st.columns([1, 1, 2])
    with relation_controls[0]:
        x_feature = st.selectbox(
            "Horizontal axis",
            FEATURES,
            index=FEATURES.index("rainfall"),
            format_func=lambda value: FEATURE_LABELS[value],
        )
    with relation_controls[1]:
        y_feature = st.selectbox(
            "Vertical axis",
            FEATURES,
            index=FEATURES.index("humidity"),
            format_func=lambda value: FEATURE_LABELS[value],
        )
    crops = sorted(dataset["Crop"].unique())
    default_crops = ["Rice", "Maize", "Cotton", "Coffee", "Banana", "Apple"]
    with relation_controls[2]:
        selected_crops = st.multiselect(
            "Crop classes (up to 8)",
            crops,
            default=default_crops,
            max_selections=8,
        )
    scatter_data = dataset[dataset["Crop"].isin(selected_crops)]
    if selected_crops:
        scatter_chart = (
            alt.Chart(scatter_data)
            .mark_circle(size=66, opacity=.7, stroke="#080d0a", strokeWidth=.5)
            .encode(
                x=alt.X(f"{x_feature}:Q", title=FEATURE_LABELS[x_feature], scale=alt.Scale(zero=False)),
                y=alt.Y(f"{y_feature}:Q", title=FEATURE_LABELS[y_feature], scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "Crop:N",
                    scale=alt.Scale(domain=selected_crops, range=CHART_COLORS[:len(selected_crops)]),
                    title="Crop",
                ),
                tooltip=[
                    alt.Tooltip("Crop:N"),
                    alt.Tooltip(f"{x_feature}:Q", title=FEATURE_LABELS[x_feature], format=".2f"),
                    alt.Tooltip(f"{y_feature}:Q", title=FEATURE_LABELS[y_feature], format=".2f"),
                ],
            )
        )
        st.altair_chart(style_chart(scatter_chart, 440), width="stretch")
    else:
        st.info("Select at least one crop class to draw the relationship map.")

    section_heading("Crop signature heatmap", "Normalized crop averages reveal unusually high and low requirements at a glance.")
    crop_means = dataset.groupby("Crop", as_index=True)[FEATURES].mean()
    crop_profiles = (crop_means - crop_means.mean()) / crop_means.std(ddof=0)
    profile_long = (
        crop_profiles.rename(columns=FEATURE_LABELS)
        .reset_index()
        .melt(id_vars="Crop", var_name="Feature", value_name="Relative level")
    )
    profile_base = alt.Chart(profile_long).encode(
        x=alt.X("Feature:N", title=None, sort=list(FEATURE_LABELS.values())),
        y=alt.Y("Crop:N", title=None),
    )
    profile_heat = profile_base.mark_rect(cornerRadius=2).encode(
        color=alt.Color(
            "Relative level:Q",
            scale=alt.Scale(domain=[-2.2, 0, 2.2], range=["#4f7cff", "#18251c", "#f0c75e"]),
            title="Relative level",
        ),
        tooltip=[
            alt.Tooltip("Crop:N"),
            alt.Tooltip("Feature:N"),
            alt.Tooltip("Relative level:Q", format="+.2f"),
        ],
    )
    profile_text = profile_base.mark_text(fontSize=9).encode(
        text=alt.Text("Relative level:Q", format="+.1f"),
        color=alt.condition("abs(datum['Relative level']) > 1.15", alt.value("#071109"), alt.value("#c8d6cb")),
    )
    st.altair_chart(style_chart(profile_heat + profile_text, 560), width="stretch")
    st.caption("Values are z-scores across crop averages: blue is below the dataset norm, amber is above it.")

    stat_col, corr_col = st.columns([.78, 1.22], gap="large")
    with stat_col:
        section_heading("Feature summary", "Central tendency and spread for each signal.")
        summary = pd.DataFrame(eda["statistics"]).T
        summary.index = [FEATURE_LABELS.get(value, value) for value in summary.index]
        summary.index.name = "Feature"
        st.dataframe(summary.style.format("{:.2f}"), width="stretch", height=326)
    with corr_col:
        section_heading("Correlation heatmap", "Green indicates positive relationships; coral indicates negative ones.")
        correlation_frame = dataset[FEATURES].corr()
        correlation_long = (
            correlation_frame.rename(index=FEATURE_LABELS, columns=FEATURE_LABELS)
            .rename_axis("Feature A")
            .reset_index()
            .melt(id_vars="Feature A", var_name="Feature B", value_name="Correlation")
        )
        correlation_base = alt.Chart(correlation_long).encode(
            x=alt.X("Feature A:N", title=None, sort=list(FEATURE_LABELS.values())),
            y=alt.Y("Feature B:N", title=None, sort=list(FEATURE_LABELS.values())),
        )
        correlation_heat = correlation_base.mark_rect(cornerRadius=3).encode(
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(domain=[-1, 0, 1], range=["#ff7b72", "#18251c", "#7ee787"]),
                title="Correlation",
            ),
            tooltip=[
                alt.Tooltip("Feature A:N"),
                alt.Tooltip("Feature B:N"),
                alt.Tooltip("Correlation:Q", format=".2f"),
            ],
        )
        correlation_text = correlation_base.mark_text(fontSize=11).encode(
            text=alt.Text("Correlation:Q", format=".2f"),
            color=alt.condition("abs(datum.Correlation) > .55", alt.value("#071109"), alt.value("#dbe8de")),
        )
        st.altair_chart(style_chart(correlation_heat + correlation_text, 326), width="stretch")

    st.markdown(
        '<div class="notice">EDA describes this dataset, not universal agronomic rules. Relationships can change across regions, seasons, and measurement practices.</div>',
        unsafe_allow_html=True,
    )

else:
    render_topline("Jupyter notebook")
    render_hero(
        "Reproducible research / NB-01",
        "The complete workflow.",
        "Cell by cell.",
        "Read the project as an executable notebook, inspect the full application source, or review the model-training pipeline without leaving the dashboard.",
        ["Jupyter format", "Corrected ML workflow", "Full application", "Production inference"],
    )

    if not NOTEBOOK_PATH.exists():
        st.error("The project notebook is missing.")
        st.stop()

    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    notebook = json.loads(notebook_text)
    code_cells = sum(cell.get("cell_type") == "code" for cell in notebook.get("cells", []))
    markdown_cells = sum(cell.get("cell_type") == "markdown" for cell in notebook.get("cells", []))
    metric_cards(
        [
            ("Notebook cells", str(len(notebook.get("cells", []))), "complete walkthrough", "#f0c75e"),
            ("Code cells", str(code_cells), "executable Python", "#7ee787"),
            ("Markdown cells", str(markdown_cells), "guided explanation", "#c8f169"),
            ("Kernel", "Python 3", "Jupyter compatible", "#8bbdff"),
        ]
    )
    st.markdown(
        """
        <div class="notebook-intro">
            <div class="notebook-logo">JPy</div>
            <div><div class="notebook-title">kshetra_sense.ipynb</div>
            <div class="notebook-copy">A portable notebook containing EDA, preprocessing, model tuning, evaluation, persistence, and inference.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    notebook_tab, app_tab, training_tab, core_tab = st.tabs(
        ["Notebook walkthrough", "Full app.py", "Training pipeline", "Inference core"]
    )
    with notebook_tab:
        st.download_button(
            "Download .ipynb",
            data=notebook_text,
            file_name="kshetra_sense.ipynb",
            mime="application/x-ipynb+json",
            width="stretch",
        )
        render_notebook_cells(notebook)

    with app_tab:
        section_heading("Complete Streamlit application", "The exact source used to render this deployed interface.")
        st.code(
            Path(__file__).read_text(encoding="utf-8"),
            language="python",
            line_numbers=True,
            height=720,
        )

    with training_tab:
        section_heading("Complete training program", "Model search, evaluation, and artifact generation used by KshetraSense.")
        if TRAINING_SCRIPT_PATH.exists():
            st.code(
                TRAINING_SCRIPT_PATH.read_text(encoding="utf-8"),
                language="python",
                line_numbers=True,
                height=720,
            )
        else:
            st.error("The training script is missing.")

    with core_tab:
        section_heading("Production inference and safeguards", "Artifact verification, OOD detection, calibrated ranking, explanations, and external validation.")
        if CORE_PATH.exists():
            st.code(CORE_PATH.read_text(encoding="utf-8"), language="python", line_numbers=True, height=720)
        else:
            st.error("The inference module is missing.")
