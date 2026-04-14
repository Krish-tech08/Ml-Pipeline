import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ML Pipeline · Healthcare Expenditure vs GDP",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── MASTER CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:           #08090d;
    --bg2:          #0f1117;
    --surface:      #141720;
    --surface2:     #1c2030;
    --surface3:     #222840;
    --border:       #2a3050;
    --border2:      #344070;
    --accent:       #3b82f6;
    --accent-dim:   rgba(59,130,246,0.15);
    --accent-glow:  rgba(59,130,246,0.35);
    --green:        #10b981;
    --green-dim:    rgba(16,185,129,0.15);
    --amber:        #f59e0b;
    --amber-dim:    rgba(245,158,11,0.15);
    --red:          #ef4444;
    --red-dim:      rgba(239,68,68,0.15);
    --text:         #f0f4ff;
    --text2:        #94a3c0;
    --text3:        #4a5578;
    --pipeline-h:   90px;
    --header-h:     80px;
}

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .block-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Hide Streamlit chrome */
[data-testid="stHeader"]       { display: none !important; }
[data-testid="stToolbar"]      { display: none !important; }
[data-testid="stDecoration"]   { display: none !important; }
footer                         { display: none !important; }
#MainMenu                      { display: none !important; }

/* Remove top padding from main container */
.main .block-container {
    padding-top: calc(var(--pipeline-h) + var(--header-h) + 24px) !important;
    padding-left: 32px !important;
    padding-right: 32px !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* ── STICKY HEADER WRAPPER ── */
.sticky-top {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 9999;
}

/* ── APP HEADER ── */
.app-header {
    height: var(--header-h);
    background: linear-gradient(135deg, #0d1a3a 0%, #091428 40%, #050f20 100%);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 32px;
    gap: 20px;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 100% at 80% 50%, rgba(59,130,246,0.08), transparent);
    pointer-events: none;
}
.app-header-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, var(--accent), #6366f1);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: 0 0 20px rgba(59,130,246,0.4);
    flex-shrink: 0;
}
.app-header-text h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 20px !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    margin: 0 !important;
    letter-spacing: -0.3px;
    line-height: 1.2;
}
.app-header-text p {
    font-size: 11px !important;
    color: var(--text2) !important;
    margin: 2px 0 0 !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 500;
}
.app-header-badges {
    margin-left: auto;
    display: flex;
    gap: 8px;
    flex-shrink: 0;
}
.header-badge {
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
    border: 1px solid;
}
.badge-blue  { background: var(--accent-dim);  border-color: var(--accent); color: #93c5fd; }
.badge-green { background: var(--green-dim);   border-color: var(--green);  color: #6ee7b7; }
.badge-amber { background: var(--amber-dim);   border-color: var(--amber);  color: #fcd34d; }

/* ── PIPELINE BAR ── */
.pipeline-bar {
    height: var(--pipeline-h);
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 20px;
    gap: 0;
    overflow: hidden;
}

/* ── STEP NODE ── */
.step-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 5px;
    position: relative;
    cursor: pointer;
    flex-shrink: 0;
    padding: 0 6px;
    min-width: 86px;
    transition: opacity 0.2s;
}
.step-node:hover { opacity: 0.85; }

.step-circle {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700;
    border: 2px solid;
    transition: all 0.3s ease;
    position: relative;
}

/* States */
.step-done .step-circle {
    background: var(--green-dim);
    border-color: var(--green);
    color: var(--green);
}
.step-done .step-circle::after {
    content: '✓';
    font-size: 14px;
    font-weight: 700;
}
.step-active .step-circle {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
    box-shadow: 0 0 18px var(--accent-glow), 0 0 40px rgba(59,130,246,0.2);
    animation: pulse-ring 2s ease-in-out infinite;
}
.step-pending .step-circle {
    background: transparent;
    border-color: var(--border2);
    color: var(--text3);
}

@keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0 var(--accent-glow), 0 0 18px var(--accent-glow); }
    70%  { box-shadow: 0 0 0 8px rgba(59,130,246,0), 0 0 18px var(--accent-glow); }
    100% { box-shadow: 0 0 0 0 rgba(59,130,246,0), 0 0 18px var(--accent-glow); }
}

.step-label {
    font-size: 9px;
    font-weight: 600;
    text-align: center;
    line-height: 1.3;
    letter-spacing: 0.3px;
    white-space: pre-line;
}
.step-done   .step-label { color: var(--green); }
.step-active .step-label { color: var(--accent); }
.step-pending .step-label { color: var(--text3); }

/* Active step underline */
.step-active::after {
    content: '';
    position: absolute;
    bottom: -var(--pipeline-h);
    left: 50%; transform: translateX(-50%);
    width: 28px; height: 2px;
    background: var(--accent);
    border-radius: 2px;
    bottom: -4px;
}

/* ── CONNECTOR LINE ── */
.step-connector {
    flex: 1;
    height: 2px;
    position: relative;
    overflow: visible;
    min-width: 8px;
    max-width: 30px;
    margin-bottom: 18px;
}
.step-connector-fill {
    height: 2px;
    width: 100%;
    border-radius: 1px;
}
.conn-done    { background: linear-gradient(90deg, var(--green), var(--green)); }
.conn-partial { background: linear-gradient(90deg, var(--green), var(--border2)); }
.conn-pending { background: var(--border); }

/* ── STEP NUMBER BADGE in circle ── */
.step-num { font-family: 'JetBrains Mono', monospace; }

/* ── SECTION HEADER ── */
.sec-header {
    margin-bottom: 28px;
    padding: 24px 28px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    position: relative;
    overflow: hidden;
}
.sec-header::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, var(--accent), var(--green));
    border-radius: 2px 0 0 2px;
}
.sec-header-inner {
    display: flex; align-items: center; gap: 16px;
}
.sec-num-badge {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, var(--accent), #6366f1);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif;
    font-weight: 800; font-size: 18px; color: #fff;
    flex-shrink: 0;
    box-shadow: 0 0 16px var(--accent-glow);
}
.sec-header h2 {
    font-family: 'Syne', sans-serif !important;
    font-size: 22px !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    margin: 0 0 4px !important;
}
.sec-header p {
    font-size: 13px !important;
    color: var(--text2) !important;
    margin: 0 !important;
}

/* ── CARD ── */
.ml-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.ml-card.accent-left  { border-left: 3px solid var(--accent); }
.ml-card.green-left   { border-left: 3px solid var(--green); }
.ml-card.amber-left   { border-left: 3px solid var(--amber); }
.card-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 16px;
}

/* ── BUTTONS ── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover {
    background: #2563eb !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── FORM ELEMENTS ── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stSelectbox > div > div:hover,
.stMultiSelect > div > div:hover {
    border-color: var(--accent) !important;
}
.stSelectbox label, .stMultiSelect label, .stNumberInput label,
.stSlider label, .stRadio label, .stCheckbox label,
.stFileUploader label {
    color: var(--text2) !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
p, span, div { color: var(--text); }
h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }

/* Dropdown menu */
[data-baseweb="popover"] ul,
[data-baseweb="menu"] ul {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
}
[data-baseweb="popover"] ul li,
[data-baseweb="menu"] ul li {
    color: var(--text) !important;
}
[data-baseweb="popover"] ul li:hover,
[data-baseweb="menu"] ul li:hover {
    background: var(--accent-dim) !important;
}

/* Multiselect tags */
.stMultiSelect span[data-baseweb="tag"] {
    background: var(--accent-dim) !important;
    color: #93c5fd !important;
}

/* ── SLIDER ── */
.stSlider > div > div > div {
    background: var(--accent) !important;
}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {
    color: var(--text2) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 2px dashed var(--border2) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: var(--text2) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }
[data-testid="stDataFrame"] th {
    background: var(--surface2) !important;
    color: var(--text2) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stDataFrame"] td {
    color: var(--text) !important;
    background: var(--surface) !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text3) !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { color: var(--text2) !important; }

/* ── ALERTS ── */
.stSuccess > div {
    background: var(--green-dim) !important;
    border: 1px solid var(--green) !important;
    border-radius: 8px !important;
    color: #6ee7b7 !important;
}
.stWarning > div {
    background: var(--amber-dim) !important;
    border: 1px solid var(--amber) !important;
    border-radius: 8px !important;
    color: #fcd34d !important;
}
.stError > div {
    background: var(--red-dim) !important;
    border: 1px solid var(--red) !important;
    border-radius: 8px !important;
    color: #fca5a5 !important;
}
.stInfo > div {
    background: var(--accent-dim) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 8px !important;
    color: #93c5fd !important;
}
.stSuccess p, .stSuccess span,
.stWarning p, .stWarning span,
.stError p, .stError span,
.stInfo p, .stInfo span { color: inherit !important; }

/* ── RADIO ── */
.stRadio > div { flex-direction: row !important; gap: 10px !important; flex-wrap: wrap !important; }
.stRadio > div > label {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    color: var(--text2) !important;
}
.stRadio > div > label:hover {
    border-color: var(--accent) !important;
    background: var(--accent-dim) !important;
    color: var(--text) !important;
}
.stRadio label p, .stRadio label span { color: inherit !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px 10px 0 0 !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text3) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background: var(--surface2) !important;
    border-bottom: 2px solid var(--accent) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 20px !important;
}

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.streamlit-expanderHeader p, .streamlit-expanderHeader span { color: var(--text) !important; }
.streamlit-expanderContent {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* ── JSON ── */
.stJson {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── PLOTLY ── */
[data-testid="stPlotlyChart"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── CHECKBOX ── */
.stCheckbox label, .stCheckbox span { color: var(--text2) !important; }

/* ── SPINNER ── */
.stSpinner > div { border-color: var(--accent) !important; }

hr { border-color: var(--border) !important; }

/* ── FEATURE PILL ── */
.feat-pill {
    background: var(--accent-dim);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 6px;
    padding: 7px 12px;
    margin-bottom: 6px;
    font-size: 13px;
    color: #93c5fd;
    font-weight: 500;
    display: flex; align-items: center; gap: 8px;
}
.feat-pill-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
}

/* ── MODEL CARD ── */
.model-card {
    background: var(--surface2);
    border: 2px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin-top: -8px;
    transition: all 0.25s ease;
    font-size: 12px;
    line-height: 1.5;
}
.model-card.active {
    background: var(--accent-dim);
    border-color: var(--accent);
    color: #93c5fd;
}
.model-card.inactive { color: var(--text3); }

/* ── INFO PANEL ── */
.info-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
}
.info-panel-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 16px;
}
.info-item {
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
    font-size: 12px;
    line-height: 1.6;
}
.info-item b { font-weight: 700; }
.info-blue  { background: var(--accent-dim);  border: 1px solid rgba(59,130,246,0.25); }
.info-blue b { color: #93c5fd; }
.info-amber { background: var(--amber-dim);   border: 1px solid rgba(245,158,11,0.25); }
.info-amber b { color: #fcd34d; }
.info-green { background: var(--green-dim);   border: 1px solid rgba(16,185,129,0.25); }
.info-green b { color: #6ee7b7; }
.info-blue span  { color: #93c5fd; }
.info-amber span { color: #fcd34d; }
.info-green span { color: #6ee7b7; }

/* ── PROGRESS BAR OVERRIDE ── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent), var(--green)) !important;
    border-radius: 4px !important;
}
.stProgress > div {
    background: var(--surface2) !important;
    border-radius: 4px !important;
}

/* ── STEP CONTENT TRANSITION ── */
.step-content-enter {
    animation: slideUp 0.35s cubic-bezier(0.16, 1, 0.3, 1) both;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ─────────────────────────────────────────────────────
defaults = dict(
    step=0, problem_type=None, df=None, target=None, features=None,
    df_clean=None, selected_features=None, X_train=None, X_test=None,
    y_train=None, y_test=None, model=None, model_name=None,
    k_folds=5, trained_model=None, cv_scores=None,
    test_size=0.2, random_state=42, outlier_indices=[]
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Plotly dark theme ──────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(20,23,32,1)',
    plot_bgcolor='rgba(28,32,48,1)',
    font_color='#94a3c0',
    font_family='Space Grotesk',
)
COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#f43f5e"]


# ══════════════════════════════════════════════════════════════════════════════
# STICKY HEADER + PIPELINE BAR
# ══════════════════════════════════════════════════════════════════════════════
STEPS = [
    ("Problem\nType",         "Define ML task:\nclassification or regression"),
    ("Data\nInput",           "Upload & preview\nhealthcare dataset"),
    ("EDA",                   "Explore distributions,\ncorrelations & missing data"),
    ("Engineering\n& Cleaning","Impute missing values,\ndetect & remove outliers"),
    ("Feature\nSelection",    "Choose the most\npredictive variables"),
    ("Data\nSplit",           "Train / test partition\nwith optional stratification"),
    ("Model\nSelection",      "Pick ML algorithm\nbest-suited to the task"),
    ("Training &\nValidation","K-fold cross-validation\n& model fitting"),
    ("Performance\nMetrics",  "Evaluate accuracy,\nR², RMSE & confusion matrix"),
    ("Hyper-\nParam Tuning",  "Grid / random search\nfor optimal hyperparameters"),
]

def render_sticky_header():
    current = st.session_state.step
    # Build step nodes HTML
    nodes_html = ""
    for i, (label, tooltip) in enumerate(STEPS):
        done   = i < current
        active = i == current
        # Circle content
        if done:
            circle_inner = "✓"
            state_class  = "step-done"
            num_display  = ""
        else:
            circle_inner = f'<span class="step-num">{i+1}</span>'
            state_class  = "step-active" if active else "step-pending"
            num_display  = ""

        # Connector after (not after last)
        connector_html = ""
        if i < len(STEPS) - 1:
            if done:
                conn_cls = "conn-done"
            elif active:
                conn_cls = "conn-partial"
            else:
                conn_cls = "conn-pending"
            connector_html = f'<div class="step-connector"><div class="step-connector-fill {conn_cls}"></div></div>'

        nodes_html += f"""
        <div class="step-node {state_class}" title="{tooltip}">
            <div class="step-circle">{circle_inner}</div>
            <span class="step-label">{label}</span>
        </div>
        {connector_html}
        """

    progress_pct = int((current / (len(STEPS) - 1)) * 100)

    st.markdown(f"""
    <div class="sticky-top">
        <!-- App Header -->
        <div class="app-header">
            <div class="app-header-icon">🏥</div>
            <div class="app-header-text">
                <h1>ML Pipeline Studio</h1>
                <p>Healthcare Expenditure vs GDP Analysis</p>
            </div>
            <div class="app-header-badges" style="display:flex;align-items:center;gap:10px;">
                <span class="header-badge badge-blue">💊 Clinical Finance</span>
                <span class="header-badge badge-green">📊 GDP Analytics</span>
                <span class="header-badge badge-amber">Step {current+1} of {len(STEPS)}</span>
            </div>
        </div>
        <!-- Pipeline Bar -->
        <div class="pipeline-bar">
            {nodes_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Section header ─────────────────────────────────────────────────────────────
def section_header(step_n, title, subtitle=""):
    st.markdown(f"""
    <div class="sec-header step-content-enter">
        <div class="sec-header-inner">
            <div class="sec-num-badge">{step_n:02d}</div>
            <div>
                <h2>{title}</h2>
                {'<p>' + subtitle + '</p>' if subtitle else ''}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Card helper ────────────────────────────────────────────────────────────────
def card(content_fn, title="", accent_class="accent-left"):
    st.markdown(f"""
    <div class="ml-card {accent_class}">
        {'<div class="card-label">' + title + '</div>' if title else ''}
    """, unsafe_allow_html=True)
    content_fn()
    st.markdown("</div>", unsafe_allow_html=True)


# ── Render sticky header (always) ─────────────────────────────────────────────
render_sticky_header()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — PROBLEM TYPE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 0:
    section_header(1, "Problem Type", "Define the machine learning task for Healthcare Expenditure vs GDP analysis")

    st.markdown('<div class="step-content-enter">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        choice = st.radio("", ["Classification", "Regression"],
                          horizontal=True, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            active_cls = "#3b82f6" if choice == "Classification" else "#2a3050"
            bg_cls = "rgba(59,130,246,0.1)" if choice == "Classification" else "#1c2030"
            text_col = "#93c5fd" if choice == "Classification" else "#4a5578"
            st.markdown(f"""
            <div style="background:{bg_cls};border:2px solid {active_cls};border-radius:12px;
                        padding:24px;text-align:center;transition:all 0.3s;">
                <div style="width:48px;height:48px;background:linear-gradient(135deg,#3b82f6,#6366f1);border-radius:10px;
                            display:flex;align-items:center;justify-content:center;
                            font-size:22px;margin:0 auto 12px;box-shadow:0 0 20px rgba(59,130,246,0.3);">🎯</div>
                <div style="font-family:'Syne',sans-serif;font-weight:800;color:{text_col};font-size:16px;margin-bottom:8px;">Classification</div>
                <div style="color:#4a5578;font-size:12px;line-height:1.6;">
                    Predict discrete outcomes such as high/low cost risk categories
                </div>
                <div style="margin-top:14px;font-size:10px;color:#3b82f6;font-weight:700;letter-spacing:1px;">
                    LOGISTIC · SVM · RANDOM FOREST
                </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            active_reg = "#10b981" if choice == "Regression" else "#2a3050"
            bg_reg = "rgba(16,185,129,0.1)" if choice == "Regression" else "#1c2030"
            text_reg = "#6ee7b7" if choice == "Regression" else "#4a5578"
            st.markdown(f"""
            <div style="background:{bg_reg};border:2px solid {active_reg};border-radius:12px;
                        padding:24px;text-align:center;transition:all 0.3s;">
                <div style="width:48px;height:48px;background:linear-gradient(135deg,#10b981,#06b6d4);border-radius:10px;
                            display:flex;align-items:center;justify-content:center;
                            font-size:22px;margin:0 auto 12px;box-shadow:0 0 20px rgba(16,185,129,0.3);">📈</div>
                <div style="font-family:'Syne',sans-serif;font-weight:800;color:{text_reg};font-size:16px;margin-bottom:8px;">Regression</div>
                <div style="color:#4a5578;font-size:12px;line-height:1.6;">
                    Forecast continuous values like per-capita health expenditure
                </div>
                <div style="margin-top:14px;font-size:10px;color:#10b981;font-weight:700;letter-spacing:1px;">
                    LINEAR · SVR · RANDOM FOREST
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Continue →", use_container_width=True):
            st.session_state.problem_type = choice
            st.session_state.step = 1
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA INPUT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 1:
    section_header(2, "Data Input", "Upload the Healthcare Expenditure vs GDP dataset and configure the prediction target")

    st.markdown('<div class="step-content-enter">', unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
        <div class="ml-card accent-left">
            <div class="card-label">Upload Dataset</div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop your CSV / Excel file — Healthcare Expenditure vs GDP",
            type=["csv", "xlsx", "xls"],
            help="Healthcare Financing dataset (OWID) — che_gdp, che_pc_usd, gghed_che etc."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded:
            try:
                if uploaded.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded)
                else:
                    df = pd.read_csv(uploaded)

                st.session_state.df = df

                n_num  = df.select_dtypes(include=np.number).shape[1]
                n_cat  = df.select_dtypes(include='object').shape[1]
                n_miss = df.isnull().sum().sum()

                m1, m2, m3, m4 = st.columns(4)
                for col_m, val, lbl in zip([m1, m2, m3, m4],
                                           [df.shape[0], df.shape[1], n_num, n_miss],
                                           ["Rows", "Columns", "Numeric", "Missing"]):
                    with col_m:
                        st.metric(lbl, f"{val:,}")

                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📋 Preview data (first 20 rows)"):
                    st.dataframe(df.head(20), use_container_width=True, height=300)

                st.markdown('<div class="card-label" style="margin-top:16px;">Target Variable</div>', unsafe_allow_html=True)
                target = st.selectbox("Select the column to predict", df.columns.tolist())
                st.session_state.target = target

                other_cols = [c for c in df.columns if c != target]
                st.markdown('<div class="card-label" style="margin-top:16px;">Input Features</div>', unsafe_allow_html=True)
                features = st.multiselect("Select features (default = all)", other_cols, default=other_cols)
                st.session_state.features = features

                if len(features) >= 2:
                    st.markdown('<div class="card-label" style="margin-top:16px;">PCA Preview</div>', unsafe_allow_html=True)
                    pca_dims = st.radio("PCA dimensions", [2, 3], horizontal=True)

                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler

                    df_sub = df[features + [target]].dropna()
                    X_pca = df_sub[features].select_dtypes(include=np.number)
                    if X_pca.shape[1] >= 2:
                        scaler = StandardScaler()
                        Xs = scaler.fit_transform(X_pca.fillna(0))
                        n_comp = min(pca_dims, X_pca.shape[1])
                        pca = PCA(n_components=n_comp)
                        comp = pca.fit_transform(Xs)

                        y_col = df_sub[target].astype(str)
                        if n_comp == 2:
                            fig = px.scatter(x=comp[:,0], y=comp[:,1], color=y_col,
                                             labels={"x":"PC1","y":"PC2"},
                                             template="plotly_dark",
                                             color_discrete_sequence=COLORS)
                        else:
                            fig = px.scatter_3d(x=comp[:,0], y=comp[:,1], z=comp[:,2],
                                                color=y_col,
                                                labels={"x":"PC1","y":"PC2","z":"PC3"},
                                                template="plotly_dark",
                                                color_discrete_sequence=COLORS)
                            fig.update_traces(marker_size=3)

                        ev = pca.explained_variance_ratio_
                        title_ev = " | ".join([f"PC{i+1}: {v*100:.1f}%" for i, v in enumerate(ev)])
                        fig.update_layout(
                            height=400, margin=dict(l=0,r=0,t=40,b=0),
                            **PLOTLY_LAYOUT,
                            title=dict(text=f"PCA — {title_ev}", font=dict(color='#3b82f6', size=13)),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                if st.button("Continue to EDA →", use_container_width=True):
                    st.session_state.df_clean = df[features + [target]].copy()
                    st.session_state.step = 2
                    st.rerun()

            except Exception as e:
                st.error(f"Error loading file: {e}")

    with col_right:
        st.markdown("""
        <div class="info-panel">
            <div class="info-panel-title">Dataset Guide</div>
            <div class="info-item info-blue">
                <b>📂 Healthcare Expenditure vs GDP</b><br>
                <span>OWID dataset by Ortiz-Ospina &amp; Roser — global health expenditure indicators</span>
            </div>
            <div class="info-item info-amber">
                <b>🎯 Recommended targets</b><br>
                <span>che_gdp · che_pc_usd · gghed_che</span>
            </div>
            <div class="info-item info-green">
                <b>💡 Tip</b><br>
                <span>Drop identifier columns like 'country' before modeling to avoid data leakage.</span>
            </div>
            <hr style="border-color:#2a3050;margin:14px 0;">
            <div style="font-size:11px;color:#4a5578;">Supported: CSV · XLSX · XLS</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back", key="back1"):
            st.session_state.step = 0
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    section_header(3, "Exploratory Data Analysis", "Understand distributions, correlations and missing patterns in the healthcare expenditure data")

    df = st.session_state.df_clean
    target = st.session_state.target

    st.markdown('<div class="step-content-enter">', unsafe_allow_html=True)
    if df is None:
        st.warning("No data found. Go back to Step 2.")
    else:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols     = df.select_dtypes(include='object').columns.tolist()

        with st.expander("📊 Descriptive Statistics", expanded=True):
            st.dataframe(df.describe().T, use_container_width=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📉 Distributions", "🔗 Correlation", "❓ Missing Data", "🎯 Target Analysis"])

        with tab1:
            cols_to_plot = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:6])
            if cols_to_plot:
                n_cols = 3
                n_rows = -(-len(cols_to_plot) // n_cols)
                fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols_to_plot)
                for idx, col in enumerate(cols_to_plot):
                    r, c = divmod(idx, n_cols)
                    fig.add_trace(go.Histogram(x=df[col].dropna(), name=col,
                                               marker_color=COLORS[idx % len(COLORS)],
                                               showlegend=False), row=r+1, col=c+1)
                fig.update_layout(height=300*n_rows, **PLOTLY_LAYOUT, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                template="plotly_dark")
                fig.update_layout(height=600, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

                if target in numeric_cols:
                    top_corr = corr[target].drop(target).abs().sort_values(ascending=False).head(10)
                    fig2 = px.bar(x=top_corr.values, y=top_corr.index, orientation='h',
                                  color=top_corr.values,
                                  color_continuous_scale=['#ef4444','#f59e0b','#3b82f6'],
                                  template="plotly_dark",
                                  labels={"x":"Absolute Correlation","y":""})
                    fig2.update_layout(height=350, **PLOTLY_LAYOUT,
                                       title=dict(text="Top correlations with target", font=dict(color='#f0f4ff')))
                    st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            miss = df.isnull().sum()
            miss = miss[miss > 0].sort_values(ascending=False)
            if miss.empty:
                st.success("✅ No missing values detected!")
            else:
                fig = px.bar(x=miss.index, y=miss.values,
                             color=miss.values, color_continuous_scale=["#f59e0b","#ef4444"],
                             template="plotly_dark",
                             labels={"x":"Column","y":"Missing Count"})
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"**{len(miss)} columns** have missing values · **{df.isnull().sum().sum():,}** total NaN cells")

        with tab4:
            if target in numeric_cols:
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.histogram(df, x=target, template="plotly_dark",
                                       color_discrete_sequence=["#3b82f6"])
                    fig.update_layout(**PLOTLY_LAYOUT,
                                      title=dict(text=f"Distribution of {target}", font=dict(color='#f0f4ff')))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.box(df, y=target, template="plotly_dark",
                                 color_discrete_sequence=["#10b981"])
                    fig.update_layout(**PLOTLY_LAYOUT,
                                      title=dict(text=f"Box Plot — {target}", font=dict(color='#f0f4ff')))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(df[target].value_counts(), template="plotly_dark",
                             color_discrete_sequence=["#3b82f6"])
                fig.update_layout(**PLOTLY_LAYOUT,
                                  title=dict(text=f"Class Distribution — {target}", font=dict(color='#f0f4ff')))
                st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.step = 1; st.rerun()
        with col2:
            if st.button("Continue to Data Engineering →", use_container_width=True):
                st.session_state.step = 3; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DATA ENGINEERING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    section_header(4, "Data Engineering & Cleaning", "Handle missing values and detect/remove outliers in the healthcare dataset")

    df = st.session_state.df_clean.copy()
    target = st.session_state.target
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]

    st.markdown('<div class="step-content-enter">', unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="ml-card accent-left"><div class="card-label">Missing Value Strategy</div>', unsafe_allow_html=True)
        impute_method = st.selectbox("Imputation method",
                                     ["None (drop rows)", "Mean", "Median", "Mode", "Forward Fill", "Zero"])
        if st.button("Apply Imputation"):
            if impute_method == "None (drop rows)":
                df = df.dropna()
            elif impute_method == "Mean":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif impute_method == "Median":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif impute_method == "Mode":
                for c in df.columns:
                    df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else np.nan)
            elif impute_method == "Forward Fill":
                df = df.ffill()
            elif impute_method == "Zero":
                df = df.fillna(0)
            st.session_state.df_clean = df
            st.success(f"✅ Imputation applied ({impute_method}). Rows remaining: {len(df):,}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="ml-card green-left"><div class="card-label">Outlier Detection</div>', unsafe_allow_html=True)
        outlier_method = st.selectbox("Detection method",
                                      ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])

        feat_for_outlier = st.multiselect("Features to use for outlier detection",
                                           numeric_cols, default=numeric_cols[:4])

        if st.button("Detect Outliers") and feat_for_outlier:
            df_sub = df[feat_for_outlier].dropna()

            if outlier_method == "IQR":
                mask = pd.Series(False, index=df_sub.index)
                for col in feat_for_outlier:
                    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    mask |= (df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)
                outlier_idx = df_sub.index[mask[df_sub.index]].tolist()

            elif outlier_method == "Isolation Forest":
                from sklearn.ensemble import IsolationForest
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                Xs = sc.fit_transform(df_sub.fillna(0))
                preds = IsolationForest(contamination=0.05, random_state=42).fit_predict(Xs)
                outlier_idx = df_sub.index[preds == -1].tolist()

            elif outlier_method == "DBSCAN":
                from sklearn.cluster import DBSCAN
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                Xs = sc.fit_transform(df_sub.fillna(0))
                labels = DBSCAN(eps=1.5, min_samples=5).fit_predict(Xs)
                outlier_idx = df_sub.index[labels == -1].tolist()

            elif outlier_method == "OPTICS":
                from sklearn.cluster import OPTICS
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                Xs = sc.fit_transform(df_sub.fillna(0))
                labels = OPTICS(min_samples=5).fit_predict(Xs)
                outlier_idx = df_sub.index[labels == -1].tolist()

            st.session_state.outlier_indices = outlier_idx

            pct = 100 * len(outlier_idx) / max(len(df), 1)
            if outlier_idx:
                st.warning(f"⚠️ Detected **{len(outlier_idx):,}** outliers ({pct:.1f}% of data)")

                if len(feat_for_outlier) >= 2:
                    is_out = pd.Series(df.index.isin(outlier_idx), index=df.index)
                    color_labels = is_out.map({True:"Outlier", False:"Normal"})
                    fig = px.scatter(df, x=feat_for_outlier[0], y=feat_for_outlier[1],
                                     color=color_labels,
                                     color_discrete_map={"Outlier":"#ef4444","Normal":"#3b82f6"},
                                     template="plotly_dark",
                                     title=f"Outliers via {outlier_method}")
                    fig.update_layout(**PLOTLY_LAYOUT, height=380,
                                      title=dict(text=f"Outliers via {outlier_method}", font=dict(color='#f0f4ff')))
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("👀 Preview outlier rows"):
                    st.dataframe(df.loc[outlier_idx[:50]], use_container_width=True)

                if st.button("🗑 Remove Outliers from Dataset", type="primary"):
                    df = df.drop(index=outlier_idx)
                    st.session_state.df_clean = df
                    st.session_state.outlier_indices = []
                    st.success(f"✅ Removed {len(outlier_idx)} outliers. Dataset now has {len(df):,} rows.")
                    st.rerun()
            else:
                st.success("✅ No outliers detected!")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="info-panel">
            <div class="info-panel-title">Current Data State</div>
        """, unsafe_allow_html=True)
        df_now = st.session_state.df_clean
        m1, m2 = st.columns(2)
        m1.metric("Rows", f"{len(df_now):,}")
        m2.metric("Columns", f"{df_now.shape[1]}")
        m1.metric("Missing", f"{df_now.isnull().sum().sum():,}")
        m2.metric("Outliers", f"{len(st.session_state.outlier_indices):,}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.step = 2; st.rerun()
    with col2:
        if st.button("Continue to Feature Selection →", use_container_width=True):
            st.session_state.step = 4; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    section_header(5, "Feature Selection", "Identify the most informative healthcare finance predictors for GDP analysis")

    df = st.session_state.df_clean.copy().dropna()
    target = st.session_state.target
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]

    st.markdown('<div class="step-content-enter">', unsafe_allow_html=True)
    method = st.radio("Selection method",
                      ["Variance Threshold", "Correlation with Target", "Information Gain (Mutual Info)"],
                      horizontal=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        selected = numeric_cols.copy()

        if method == "Variance Threshold":
            thresh = st.slider("Variance threshold", 0.0, 5.0, 0.1, 0.01)
            from sklearn.feature_selection import VarianceThreshold
            if numeric_cols:
                vt = VarianceThreshold(threshold=thresh)
                try:
                    vt.fit(df[numeric_cols].fillna(0))
                    mask = vt.get_support()
                    selected = [c for c, m in zip(numeric_cols, mask) if m]
                    removed  = [c for c, m in zip(numeric_cols, mask) if not m]

                    variances = df[numeric_cols].var().sort_values(ascending=False)
                    fig = px.bar(x=variances.index, y=variances.values,
                                 color=(variances >= thresh).map({True:"#3b82f6", False:"#ef4444"}),
                                 template="plotly_dark",
                                 labels={"x":"Feature","y":"Variance"})
                    fig.add_hline(y=thresh, line_dash="dash", line_color="#f59e0b",
                                  annotation_text=f"Threshold={thresh}")
                    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=380)
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(f"Keeping **{len(selected)}** features · Removing **{len(removed)}**")
                except Exception as e:
                    st.error(str(e))

        elif method == "Correlation with Target":
            if target in df.select_dtypes(include=np.number).columns:
                corr_thresh = st.slider("Min |correlation| with target", 0.0, 1.0, 0.1, 0.01)
                corrs = df[numeric_cols + [target]].corr()[target].drop(target).abs()
                selected = corrs[corrs >= corr_thresh].index.tolist()

                fig = px.bar(x=corrs.sort_values(ascending=False).index,
                             y=corrs.sort_values(ascending=False).values,
                             color=(corrs.sort_values(ascending=False) >= corr_thresh).map({True:"#3b82f6",False:"#ef4444"}),
                             template="plotly_dark",
                             labels={"x":"Feature","y":f"|Corr with {target}|"})
                fig.add_hline(y=corr_thresh, line_dash="dash", line_color="#f59e0b")
                fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=380)
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"Keeping **{len(selected)}** features above threshold")

        elif method == "Information Gain (Mutual Info)":
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            top_k = st.slider("Top K features", 1, max(1, len(numeric_cols)), min(10, len(numeric_cols)))

            X_mi = df[numeric_cols].fillna(0)
            y_mi = df[target]

            if st.session_state.problem_type == "Classification":
                try:
                    mi = mutual_info_classif(X_mi, y_mi, random_state=42)
                except:
                    mi = mutual_info_regression(X_mi, y_mi, random_state=42)
            else:
                mi = mutual_info_regression(X_mi, y_mi.fillna(0), random_state=42)

            mi_series = pd.Series(mi, index=numeric_cols).sort_values(ascending=False)
            selected = mi_series.head(top_k).index.tolist()

            fig = px.bar(x=mi_series.index, y=mi_series.values,
                         color=(mi_series.index.isin(selected)).map({True:"#3b82f6",False:"#10b981"}),
                         template="plotly_dark",
                         labels={"x":"Feature","y":"Mutual Information Score"})
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Top **{top_k}** features selected")

    with col_r:
        st.markdown("""
        <div class="info-panel">
            <div class="info-panel-title">Selected Features</div>
        """, unsafe_allow_html=True)
        for feat in selected:
            st.markdown(f"""
            <div class="feat-pill">
                <div class="feat-pill-dot"></div>
                {feat}
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.selected_features = selected

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.step = 3; st.rerun()
    with col2:
        if st.button("Continue to Data Split →", use_container_width=True):
            st.session_state.step = 5; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — DATA SPLIT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    section_header(6, "Data Split", "Partition the healthcare expenditure dataset into training and testing sets")

    df = st.session_state.df_clean.dropna()
    target   = st.session_state.target
    features = st.session_state.selected_features or [c for c in df.select_dtypes(include=np.number).columns if c != target]

    st.markdown('<div class="step-content-enter">', unsafe_allow_html=True)
    col_l, col_r = st.columns([2, 3])

    with col_l:
        test_size    = st.slider("Test set size (%)", 10, 40, 20) / 100
        random_state = st.number_input("Random seed", 0, 999, 42)
        stratify     = st.checkbox("Stratify split (classification)", value=(st.session_state.problem_type=="Classification"))

        if st.button("✂️ Split Dataset", use_container_width=True):
            from sklearn.model_selection import train_test_split
            df_sub = df[features + [target]].dropna()
            X = df_sub[features]
            y = df_sub[target]

            strat = y if (stratify and st.session_state.problem_type == "Classification") else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=strat)
                st.session_state.X_train = X_train
                st.session_state.X_test  = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test  = y_test
                st.session_state.test_size    = test_size
                st.session_state.random_state = random_state
                st.success("✅ Split successful!")
            except Exception as e:
                st.error(f"Split error: {e}")

    with col_r:
        if st.session_state.X_train is not None:
            n_train = len(st.session_state.X_train)
            n_test  = len(st.session_state.X_test)

            fig = go.Figure(data=[go.Pie(
                labels=["Train", "Test"],
                values=[n_train, n_test],
                marker=dict(colors=["#3b82f6","#10b981"]),
                hole=0.55,
                textinfo="label+percent",
                textfont_size=14
            )])
            fig.update_layout(height=320, **PLOTLY_LAYOUT, showlegend=False,
                              margin=dict(l=0,r=0,t=0,b=0),
                              annotations=[dict(text=f"{n_train+n_test:,}<br>total",
                                               x=0.5, y=0.5, showarrow=False,
                                               font=dict(size=16, color="#f0f4ff"))])
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Train samples", f"{n_train:,}")
            c2.metric("Test samples",  f"{n_test:,}")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.step = 4; st.rerun()
    with col2:
        if st.session_state.X_train is not None:
            if st.button("Continue to Model Selection →", use_container_width=True):
                st.session_state.step = 6; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — MODEL SELECTION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 6:
    section_header(7, "Model Selection", "Choose the algorithm best suited for healthcare expenditure prediction")

    pt = st.session_state.problem_type

    if pt == "Classification":
        model_options = {
            "Logistic Regression": ("🎯", "Best for binary risk classification (e.g. high/low cost)"),
            "SVM (kernel options)": ("⚡", "Effective on high-dimensional healthcare feature spaces"),
            "Random Forest": ("🌲", "Robust ensemble method for complex expenditure patterns"),
            "K-Means (unsupervised)": ("⭕", "Cluster countries/regions by healthcare spending profile"),
        }
    else:
        model_options = {
            "Linear Regression": ("📏", "Baseline forecasting of healthcare expenditure per capita"),
            "SVR (kernel options)": ("⚡", "Non-linear health cost prediction with kernel trick"),
            "Random Forest Regressor": ("🌲", "Capture complex GDP-to-health-spend relationships"),
            "K-Means (unsupervised)": ("⭕", "Segment health systems by spending patterns"),
        }

    st.markdown('<div class="step-content-enter">', unsafe_allow_html=True)
    cols = st.columns(len(model_options))
    selected_model = st.session_state.model_name or list(model_options.keys())[0]

    for i, (name, (icon, desc)) in enumerate(model_options.items()):
        with cols[i]:
            active = selected_model == name
            if st.button(f"{icon} {name}", key=f"model_{i}", use_container_width=True):
                st.session_state.model_name = name
                st.rerun()
            st.markdown(f"""
            <div class="model-card {'active' if active else 'inactive'}">
                {'<span style="color:#3b82f6;font-weight:700;">✓ Selected</span>' if active else desc}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    model_name = st.session_state.model_name
    if model_name and ("SVM" in model_name or "SVR" in model_name):
        st.markdown('<div class="card-label">SVM Kernel Options</div>', unsafe_allow_html=True)
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        st.session_state["svm_kernel"] = kernel

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.step = 5; st.rerun()
    with col2:
        if st.session_state.model_name:
            if st.button("Continue to Training →", use_container_width=True):
                st.session_state.step = 7; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — TRAINING & K-FOLD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 7:
    section_header(8, "Training & K-Fold Validation", "Train and validate your healthcare finance model with cross-validation")

    model_name = st.session_state.model_name
    pt = st.session_state.problem_type

    st.markdown('<div class="step-content-enter">', unsafe_allow_html=True)
    col_l, col_r = st.columns([2, 3])
    with col_l:
        k = st.number_input("K (number of folds)", min_value=2, max_value=20, value=5)
        st.session_state.k_folds = k

        if st.button("🚀 Train Model", use_container_width=True):
            from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
            from sklearn.preprocessing import StandardScaler, LabelEncoder

            X_train = st.session_state.X_train
            y_train = st.session_state.y_train

            if X_train is None:
                st.error("Please complete data split first.")
            else:
                Xs = StandardScaler().fit_transform(X_train.fillna(0))
                ys = y_train.fillna(0)

                le = None
                if pt == "Classification":
                    if not pd.api.types.is_numeric_dtype(ys):
                        le = LabelEncoder()
                        ys = le.fit_transform(ys.astype(str))
                    else:
                        try:
                            ys = ys.astype(int)
                        except (ValueError, TypeError):
                            le = LabelEncoder()
                            ys = le.fit_transform(ys.astype(str))

                kernel = st.session_state.get("svm_kernel", "rbf")
                try:
                    if pt == "Classification":
                        if "Logistic" in model_name:
                            from sklearn.linear_model import LogisticRegression
                            model = LogisticRegression(max_iter=500, random_state=42)
                            scoring = "accuracy"
                        elif "SVM" in model_name:
                            from sklearn.svm import SVC
                            model = SVC(kernel=kernel, random_state=42)
                            scoring = "accuracy"
                        elif "Random Forest" in model_name:
                            from sklearn.ensemble import RandomForestClassifier
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                            scoring = "accuracy"
                        else:
                            from sklearn.cluster import KMeans
                            model = KMeans(n_clusters=k, random_state=42)
                            model.fit(Xs)
                            st.session_state.trained_model = model
                            st.session_state.cv_scores = None
                            st.success("✅ K-Means clustering complete!")
                            st.rerun()

                        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
                        scores = cross_val_score(model, Xs, ys, cv=cv, scoring=scoring)
                        model.fit(Xs, ys)

                    else:
                        if "Linear" in model_name:
                            from sklearn.linear_model import LinearRegression
                            model = LinearRegression()
                        elif "SVR" in model_name:
                            from sklearn.svm import SVR
                            model = SVR(kernel=kernel)
                        elif "Random Forest" in model_name:
                            from sklearn.ensemble import RandomForestRegressor
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                        else:
                            from sklearn.cluster import KMeans
                            model = KMeans(n_clusters=k, random_state=42)
                            model.fit(Xs)
                            st.session_state.trained_model = model
                            st.session_state.cv_scores = None
                            st.success("✅ K-Means clustering complete!")
                            st.rerun()

                        cv = KFold(n_splits=k, shuffle=True, random_state=42)
                        scores = cross_val_score(model, Xs, ys, cv=cv, scoring="r2")
                        scoring = "r2"
                        model.fit(Xs, ys)

                    st.session_state.trained_model = model
                    st.session_state.cv_scores = scores
                    st.session_state["label_encoder"] = le
                    st.success(f"✅ Training complete! Mean CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")

                except Exception as e:
                    st.error(f"Training error: {e}")

    with col_r:
        cv_scores = st.session_state.cv_scores
        if cv_scores is not None:
            folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
            mean_score = np.mean(cv_scores)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=folds, y=cv_scores,
                                 marker_color=["#3b82f6" if s >= mean_score else "#10b981" for s in cv_scores],
                                 text=[f"{s:.4f}" for s in cv_scores],
                                 textposition="outside",
                                 textfont=dict(color="#f0f4ff")))
            fig.add_hline(y=mean_score, line_dash="dash", line_color="#f59e0b",
                          annotation_text=f"Mean = {mean_score:.4f}",
                          annotation_font_color="#f59e0b")
            fig.update_layout(template="plotly_dark", height=380,
                              **PLOTLY_LAYOUT,
                              title=dict(text="Cross-Validation Scores per Fold", font=dict(color='#f0f4ff')),
                              yaxis_title="Score", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Mean",      f"{cv_scores.mean():.4f}")
            c2.metric("Std Dev",   f"{cv_scores.std():.4f}")
            c3.metric("Min / Max", f"{cv_scores.min():.3f} / {cv_scores.max():.3f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.step = 6; st.rerun()
    with col2:
        if st.session_state.trained_model is not None:
            if st.button("Continue to Metrics →", use_container_width=True):
                st.session_state.step = 8; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 8:
    section_header(9, "Performance Metrics", "Evaluate model quality and assess fit for healthcare expenditure predictions")

    model   = st.session_state.trained_model
    X_train = st.session_state.X_train
    X_test  = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test  = st.session_state.y_test
    pt      = st.session_state.problem_type
    le      = st.session_state.get("label_encoder", None)

    st.markdown('<div class="step-content-enter">', unsafe_allow_html=True)
    if model is None or X_train is None:
        st.warning("Please train a model first.")
    else:
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train.fillna(0))
        X_test_s  = sc.transform(X_test.fillna(0))

        y_train_raw = y_train.fillna(0)
        y_test_raw  = y_test.fillna(0)

        if pt == "Classification" and le is not None:
            y_train_enc = le.transform(y_train_raw.astype(str))
            y_test_enc  = le.transform(y_test_raw.astype(str))
        elif pt == "Classification":
            if not pd.api.types.is_numeric_dtype(y_train_raw):
                _le = LabelEncoder()
                y_train_enc = _le.fit_transform(y_train_raw.astype(str))
                y_test_enc  = _le.transform(y_test_raw.astype(str))
            else:
                try:
                    y_train_enc = y_train_raw.astype(int)
                    y_test_enc  = y_test_raw.astype(int)
                except (ValueError, TypeError):
                    _le = LabelEncoder()
                    y_train_enc = _le.fit_transform(y_train_raw.astype(str))
                    y_test_enc  = _le.transform(y_test_raw.astype(str))
        else:
            y_train_enc = y_train_raw
            y_test_enc  = y_test_raw

        try:
            if hasattr(model, 'predict'):
                train_preds = model.predict(X_train_s)
                test_preds  = model.predict(X_test_s)
            else:
                st.info("Clustering model — no predictions available.")
                st.stop()

            if pt == "Classification":
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                train_acc = accuracy_score(y_train_enc, train_preds)
                test_acc  = accuracy_score(y_test_enc,  test_preds)
                diff = train_acc - test_acc

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Train Accuracy", f"{train_acc:.4f}")
                c2.metric("Test Accuracy",  f"{test_acc:.4f}")
                c3.metric("Gap (Train−Test)", f"{diff:.4f}",
                          delta_color="inverse" if diff > 0.05 else "normal")

                if diff > 0.1:
                    st.error("🔴 Likely **OVERFITTING** — large gap between train and test accuracy. Consider regularization or more data.")
                elif test_acc < 0.5:
                    st.warning("🟡 Likely **UNDERFITTING** — test accuracy is very low. Consider a more complex model or better features.")
                else:
                    st.success("🟢 Model appears well-fitted.")

                cm = confusion_matrix(y_test_enc, test_preds)
                fig = px.imshow(cm, text_auto=True, template="plotly_dark",
                                color_continuous_scale=["#1c2030","#3b82f6"],
                                labels=dict(x="Predicted", y="Actual"))
                fig.update_layout(**PLOTLY_LAYOUT, height=400,
                                  title=dict(text="Confusion Matrix", font=dict(color='#f0f4ff')))
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📋 Classification Report"):
                    report = classification_report(y_test_enc, test_preds, output_dict=True)
                    st.dataframe(pd.DataFrame(report).T, use_container_width=True)

            else:
                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                train_r2  = r2_score(y_train_enc, train_preds)
                test_r2   = r2_score(y_test_enc,  test_preds)
                test_rmse = np.sqrt(mean_squared_error(y_test_enc, test_preds))
                test_mae  = mean_absolute_error(y_test_enc, test_preds)
                diff = train_r2 - test_r2

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Train R²",  f"{train_r2:.4f}")
                c2.metric("Test R²",   f"{test_r2:.4f}")
                c3.metric("RMSE",      f"{test_rmse:.4f}")
                c4.metric("MAE",       f"{test_mae:.4f}")

                if diff > 0.15:
                    st.error("🔴 **OVERFITTING** — training R² much higher than test R².")
                elif test_r2 < 0.3:
                    st.warning("🟡 **UNDERFITTING** — low R² on test set.")
                else:
                    st.success("🟢 Model appears well-fitted.")

                fig = px.scatter(x=y_test_enc, y=test_preds,
                                 labels={"x":"Actual","y":"Predicted"},
                                 template="plotly_dark",
                                 color_discrete_sequence=["#3b82f6"])
                fig.add_shape(type="line",
                              x0=float(y_test_enc.min()), y0=float(y_test_enc.min()),
                              x1=float(y_test_enc.max()), y1=float(y_test_enc.max()),
                              line=dict(color="#ef4444", dash="dash"))
                fig.update_layout(**PLOTLY_LAYOUT, height=400,
                                  title=dict(text="Actual vs Predicted — Healthcare Expenditure", font=dict(color='#f0f4ff')))
                st.plotly_chart(fig, use_container_width=True)

                residuals = y_test_enc - test_preds
                fig2 = px.histogram(x=residuals, template="plotly_dark",
                                    color_discrete_sequence=["#10b981"],
                                    labels={"x":"Residual"})
                fig2.update_layout(**PLOTLY_LAYOUT, height=320,
                                   title=dict(text="Residual Distribution", font=dict(color='#f0f4ff')))
                st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Error computing metrics: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.step = 7; st.rerun()
    with col2:
        if st.button("Continue to Hyperparameter Tuning →", use_container_width=True):
            st.session_state.step = 9; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 9:
    section_header(10, "Hyperparameter Tuning", "Optimize your model for maximum predictive accuracy on healthcare expenditure data")

    model_name = st.session_state.model_name
    pt         = st.session_state.problem_type
    X_train    = st.session_state.X_train
    y_train    = st.session_state.y_train
    le         = st.session_state.get("label_encoder", None)

    st.markdown('<div class="step-content-enter">', unsafe_allow_html=True)
    if X_train is None:
        st.warning("Please complete training first.")
    else:
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train.fillna(0))
        y_train_raw = y_train.fillna(0)
        if pt == "Classification" and le is not None:
            y_enc = le.transform(y_train_raw.astype(str))
        elif pt == "Classification":
            if not pd.api.types.is_numeric_dtype(y_train_raw):
                _le2 = LabelEncoder()
                y_enc = _le2.fit_transform(y_train_raw.astype(str))
            else:
                try:
                    y_enc = y_train_raw.astype(int)
                except (ValueError, TypeError):
                    _le2 = LabelEncoder()
                    y_enc = _le2.fit_transform(y_train_raw.astype(str))
        else:
            y_enc = y_train_raw

        search_method = st.radio("Search method", ["GridSearchCV", "RandomizedSearchCV"], horizontal=True)
        cv_k = st.number_input("CV folds for tuning", 2, 10, 3)

        param_grids = {}
        if "Logistic" in (model_name or "") or "Linear Regression" in (model_name or ""):
            if "Logistic" in (model_name or ""):
                param_grids = {"C": [0.001, 0.01, 0.1, 1, 10, 100], "solver": ["lbfgs", "liblinear"], "max_iter": [100, 300, 500]}
            else:
                param_grids = {"fit_intercept": [True, False], "copy_X": [True]}
        elif "SVM" in (model_name or "") or "SVR" in (model_name or ""):
            param_grids = {"C": [0.01, 0.1, 1, 10], "kernel": ["rbf", "linear", "poly"], "gamma": ["scale", "auto"]}
        elif "Random Forest" in (model_name or ""):
            param_grids = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]}
        elif "K-Means" in (model_name or ""):
            param_grids = {"n_clusters": [2, 3, 4, 5, 6, 7, 8]}

        st.markdown('<div class="card-label" style="margin-top:16px;">Parameter Grid</div>', unsafe_allow_html=True)
        st.json(param_grids)

        n_iter = 10
        if search_method == "RandomizedSearchCV":
            n_iter = st.slider("Number of random iterations", 5, 50, 10)

        if st.button("🔍 Run Hyperparameter Search", use_container_width=True):
            if not param_grids:
                st.warning("No parameter grid defined for this model.")
            else:
                from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

                try:
                    if "Logistic" in model_name:
                        from sklearn.linear_model import LogisticRegression
                        base = LogisticRegression(random_state=42); scoring = "accuracy"
                    elif "SVM" in model_name:
                        from sklearn.svm import SVC
                        base = SVC(random_state=42); scoring = "accuracy"
                    elif "Random Forest" in model_name and pt == "Classification":
                        from sklearn.ensemble import RandomForestClassifier
                        base = RandomForestClassifier(random_state=42); scoring = "accuracy"
                    elif "Linear Regression" in model_name:
                        from sklearn.linear_model import LinearRegression
                        base = LinearRegression(); scoring = "r2"
                    elif "SVR" in model_name:
                        from sklearn.svm import SVR
                        base = SVR(); scoring = "r2"
                    elif "Random Forest" in model_name and pt == "Regression":
                        from sklearn.ensemble import RandomForestRegressor
                        base = RandomForestRegressor(random_state=42); scoring = "r2"
                    else:
                        st.warning("Hyperparameter tuning not supported for K-Means via CV.")
                        st.stop()

                    with st.spinner("Running search... this may take a moment ⏳"):
                        if search_method == "GridSearchCV":
                            searcher = GridSearchCV(base, param_grids, cv=cv_k, scoring=scoring, n_jobs=-1, refit=True)
                        else:
                            searcher = RandomizedSearchCV(base, param_grids, n_iter=n_iter, cv=cv_k,
                                                          scoring=scoring, n_jobs=-1, refit=True, random_state=42)
                        searcher.fit(X_train_s, y_enc)

                    best_params = searcher.best_params_
                    best_score  = searcher.best_score_

                    st.success(f"✅ Best {scoring}: **{best_score:.4f}**")
                    st.markdown('<div class="card-label" style="margin-top:16px;">Best Parameters</div>', unsafe_allow_html=True)
                    st.json(best_params)

                    results = pd.DataFrame(searcher.cv_results_).sort_values("rank_test_score")
                    show_cols = [c for c in results.columns if c.startswith("param_") or
                                 c in ["mean_test_score","std_test_score","rank_test_score"]]
                    st.markdown('<div class="card-label" style="margin-top:16px;">All Results</div>', unsafe_allow_html=True)
                    st.dataframe(results[show_cols].head(20), use_container_width=True)

                    fig = px.bar(results.head(15), y="mean_test_score",
                                 error_y="std_test_score",
                                 color="mean_test_score",
                                 color_continuous_scale=["#ef4444","#f59e0b","#3b82f6"],
                                 template="plotly_dark",
                                 title=f"Top 15 parameter combinations — {scoring}")
                    fig.update_layout(**PLOTLY_LAYOUT, height=400,
                                      xaxis_title="Combination #",
                                      yaxis_title=scoring, showlegend=False,
                                      title=dict(text=f"Top 15 parameter combinations — {scoring}", font=dict(color='#f0f4ff')))
                    st.plotly_chart(fig, use_container_width=True)

                    st.session_state.trained_model = searcher.best_estimator_
                    st.info("💡 The tuned model has been saved. Go back to Metrics to re-evaluate!")

                except Exception as e:
                    st.error(f"Tuning error: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Metrics"):
            st.session_state.step = 8; st.rerun()
    with col2:
        if st.button("🔄 Restart Pipeline", use_container_width=True):
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:40px 20px 24px;border-top:1px solid #2a3050;margin-top:48px;">
    <div style="display:flex;justify-content:center;align-items:center;gap:20px;margin-bottom:12px;flex-wrap:wrap;">
        <span style="font-size:12px;color:#4a5578;">💊 Clinical Cost Modeling</span>
        <span style="color:#2a3050;">|</span>
        <span style="font-size:12px;color:#4a5578;">📊 GDP Expenditure Analytics</span>
        <span style="color:#2a3050;">|</span>
        <span style="font-size:12px;color:#4a5578;">🌐 Global Health Finance</span>
    </div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#4a5578;letter-spacing:1px;">
        ML PIPELINE STUDIO · HEALTHCARE EXPENDITURE VS GDP · BUILT WITH STREAMLIT + PLOTLY
    </span>
</div>
""", unsafe_allow_html=True)
