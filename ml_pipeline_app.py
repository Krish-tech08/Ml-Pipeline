import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ML Pipeline · Healthcare Finance",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

:root {
    --bg: #f0f4f8;
    --surface: #ffffff;
    --surface2: #e8f0f7;
    --border: #c8d8e8;
    --accent: #1a6fa8;
    --accent-light: #d6eaf8;
    --accent2: #0e8c6a;
    --accent2-light: #d4f0e8;
    --accent3: #c0392b;
    --accent3-light: #fde8e6;
    --gold: #b8860b;
    --gold-light: #fdf3d0;
    --text: #1a2332;
    --text-body: #2d3e50;
    --muted: #4a6080;
    --card-bg: #ffffff;
    --header-bg: #0d3b5e;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

p, span, div, label, li, h1, h2, h3, h4, h5, h6 {
    color: var(--text);
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 6px;
    background: linear-gradient(90deg, #1a6fa8, #0e8c6a, #b8860b);
    z-index: 999;
}

[data-testid="stHeader"] { background: transparent !important; }

[data-testid="stSidebar"] {
    background: var(--header-bg) !important;
    border-right: 1px solid #1a4a6e !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    background: #155d8e !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(26,111,168,0.25) !important;
}

.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
}

.stSelectbox div[data-baseweb="select"] > div,
[data-baseweb="popover"] ul li,
[data-baseweb="menu"] ul li {
    background: var(--surface) !important;
    color: var(--text) !important;
}

.stMultiSelect span[data-baseweb="tag"] {
    background: var(--accent-light) !important;
    color: var(--accent) !important;
}
.stMultiSelect span[data-baseweb="tag"] span {
    color: var(--accent) !important;
}

.stNumberInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}

.stSlider > div > div > div { background: var(--accent) !important; }

.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"],
.stSlider .stSlider > label {
    color: var(--text-body) !important;
}

[data-testid="stFileUploader"] {
    background: var(--accent-light) !important;
    border: 2px dashed var(--accent) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: var(--text-body) !important;
}

[data-testid="stDataFrame"] { border-radius: 8px !important; }
[data-testid="stDataFrame"] th {
    background: var(--surface2) !important;
    color: var(--text) !important;
}
[data-testid="stDataFrame"] td {
    color: var(--text-body) !important;
}

.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
}
.streamlit-expanderHeader p,
.streamlit-expanderHeader span {
    color: var(--text) !important;
}
.streamlit-expanderContent {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-left: 4px solid var(--accent) !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stMetricDelta"] {
    color: var(--text-body) !important;
}

.stSuccess, .stSuccess p, .stSuccess span {
    background: var(--accent2-light) !important;
    border: 1px solid var(--accent2) !important;
    border-radius: 6px !important;
    color: #0a5c44 !important;
}
.stWarning, .stWarning p, .stWarning span {
    background: var(--gold-light) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 6px !important;
    color: #6b4a00 !important;
}
.stError, .stError p, .stError span {
    background: var(--accent3-light) !important;
    border: 1px solid var(--accent3) !important;
    border-radius: 6px !important;
    color: #7b1a13 !important;
}
.stInfo, .stInfo p, .stInfo span {
    background: var(--accent-light) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 6px !important;
    color: #0d3a5c !important;
}

.stRadio > div { flex-direction: row !important; gap: 12px !important; }
.stRadio > div > label {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 8px 18px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    color: var(--text-body) !important;
}
.stRadio > div > label:hover {
    border-color: var(--accent) !important;
    background: var(--accent-light) !important;
}
.stRadio label p,
.stRadio label span {
    color: var(--text-body) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface2) !important;
    border-radius: 8px 8px 0 0 !important;
    border-bottom: 2px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background: var(--surface) !important;
    border-bottom: 2px solid var(--accent) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 16px !important;
}

.stCheckbox label, .stCheckbox span {
    color: var(--text-body) !important;
}

.stSelectbox label, .stMultiSelect label, .stTextInput label,
.stNumberInput label, .stSlider label, .stFileUploader label,
.stCheckbox label, .stRadio label, .stTextArea label {
    color: var(--text-body) !important;
    font-weight: 500 !important;
}

.stSpinner > div {
    border-color: var(--accent) !important;
}

.stJson {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

[data-testid="stPlotlyChart"] {
    border-radius: 8px !important;
    overflow: hidden !important;
}

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ── Utility: card wrapper ──────────────────────────────────────────────────────
def card(content_fn, title="", accent="var(--accent)"):
    st.markdown(f"""
    <div style="background:var(--card-bg);border:1px solid var(--border);border-radius:10px;
                padding:24px;margin-bottom:20px;border-left:4px solid {accent};
                box-shadow:0 1px 4px rgba(26,111,168,0.08);">
        {'<p style="font-family:Inter,sans-serif;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;font-weight:600;">' + title + '</p>' if title else ''}
    """, unsafe_allow_html=True)
    content_fn()
    st.markdown("</div>", unsafe_allow_html=True)


# ── Stepper ────────────────────────────────────────────────────────────────────
def step_badge(n, label, done=False, active=False):
    if active:
        bg, col, border = "var(--accent)", "#ffffff", "var(--accent)"
    elif done:
        bg, col, border = "var(--accent2-light)", "var(--accent2)", "var(--accent2)"
    else:
        bg, col, border = "var(--surface)", "var(--muted)", "var(--border)"

    label_col = "#ffffff" if active else ("var(--accent2)" if done else "var(--muted)")

    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;gap:6px;min-width:90px;">
        <div style="width:38px;height:38px;border-radius:50%;background:{bg};border:2px solid {border};
                    display:flex;align-items:center;justify-content:center;
                    font-family:Inter,sans-serif;font-weight:700;font-size:13px;color:{col};">
            {'✓' if done else str(n)}
        </div>
        <span style="font-size:10px;color:{label_col};text-align:center;font-weight:600;line-height:1.3;letter-spacing:0.2px;">{label}</span>
    </div>"""


def connector(done=False):
    col = "var(--accent2)" if done else "var(--border)"
    return f'<div style="flex:1;height:2px;background:{col};margin-top:-14px;"></div>'


STEPS = [
    "Problem\nType", "Data\nInput", "EDA", "Engineering\n& Cleaning",
    "Feature\nSelection", "Data\nSplit", "Model\nSelection",
    "Training &\nValidation", "Metrics", "Hyper-\nParameter Tuning"
]

def render_stepper(current_step):
    html = '<div style="display:flex;align-items:flex-start;gap:0;padding:16px 10px;overflow-x:auto;">'
    for i, label in enumerate(STEPS):
        done = i < current_step
        active = i == current_step
        html += step_badge(i + 1, label, done, active)
        if i < len(STEPS) - 1:
            html += connector(done)
    html += '</div>'
    st.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;
                padding:8px;margin-bottom:28px;box-shadow:0 1px 4px rgba(26,111,168,0.06);">
        {html}
    </div>
    """, unsafe_allow_html=True)


# ── Section header ─────────────────────────────────────────────────────────────
def section_header(step_n, title, subtitle=""):
    st.markdown(f"""
    <div style="margin-bottom:24px;">
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;">
            <div style="width:36px;height:36px;border-radius:8px;
                        background:var(--accent);
                        display:flex;align-items:center;justify-content:center;
                        font-family:Inter,sans-serif;font-weight:700;font-size:13px;color:#fff;">
                {step_n:02d}
            </div>
            <h2 style="margin:0;font-size:22px;font-weight:700;color:var(--text);">{title}</h2>
        </div>
        {'<p style="color:var(--muted);margin:0 0 0 50px;font-size:14px;">' + subtitle + '</p>' if subtitle else ''}
        <div style="height:3px;background:linear-gradient(90deg,var(--accent),var(--accent2),transparent);border-radius:2px;margin-top:12px;margin-left:50px;width:200px;"></div>
    </div>
    """, unsafe_allow_html=True)


# ── Helper: check if stratify is feasible ─────────────────────────────────────
def can_stratify(y):
    """Return True only if every class has at least 2 members."""
    if not pd.api.types.is_categorical_dtype(y) and not pd.api.types.is_object_dtype(y):
        # For numeric targets, check value counts
        counts = pd.Series(y).value_counts()
    else:
        counts = pd.Series(y).value_counts()
    return bool((counts >= 2).all())


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


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0d3b5e 0%,#1a6fa8 60%,#0e8c6a 100%);
            border-radius:14px;padding:40px 40px 36px;margin-bottom:28px;
            border:1px solid #1a6fa8;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;right:0;width:300px;height:100%;
                background:url('data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 300 200\"><path d=\"M0 100 Q75 40 150 100 T300 100\" stroke=\"rgba(255,255,255,0.06)\" fill=\"none\" stroke-width=\"2\"/><path d=\"M0 130 Q75 70 150 130 T300 130\" stroke=\"rgba(255,255,255,0.04)\" fill=\"none\" stroke-width=\"2\"/><path d=\"M0 70 Q75 10 150 70 T300 70\" stroke=\"rgba(255,255,255,0.04)\" fill=\"none\" stroke-width=\"2\"/></svg>
                </div>
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px;">
        <div style="width:48px;height:48px;background:rgba(255,255,255,0.15);border-radius:10px;
                    display:flex;align-items:center;justify-content:center;font-size:24px;">🏥</div>
        <div>
            <div style="font-family:Inter,sans-serif;font-size:10px;color:rgba(255,255,255,0.7);
                        letter-spacing:3px;text-transform:uppercase;margin-bottom:4px;">
                AutoML Platform · Healthcare Finance Analytics
            </div>
            <h1 style="font-size:32px;font-weight:700;margin:0;color:#ffffff;letter-spacing:-0.5px;">
                ML Pipeline Studio
            </h1>
        </div>
    </div>
    <p style="color:rgba(255,255,255,0.8);font-size:14px;margin:0;max-width:500px;">
        End-to-end machine learning for healthcare expenditure analysis — from raw data to tuned predictive model
    </p>
    <div style="display:flex;gap:24px;margin-top:20px;flex-wrap:wrap;">
        <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);border-radius:6px;padding:8px 16px;
                    font-size:12px;color:rgba(255,255,255,0.95);font-weight:500;">
            💊 Clinical Cost Modeling
        </div>
        <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);border-radius:6px;padding:8px 16px;
                    font-size:12px;color:rgba(255,255,255,0.95);font-weight:500;">
            📊 Expenditure Forecasting
        </div>
        <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);border-radius:6px;padding:8px 16px;
                    font-size:12px;color:rgba(255,255,255,0.95);font-weight:500;">
            🌐 Global Health Metrics
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

render_stepper(st.session_state.step)

# ── Plotly theme ───────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(255,255,255,1)',
    plot_bgcolor='rgba(248,251,254,1)',
    font_color='#1a2332',
    font_family='Inter',
)
COLORS = ["#1a6fa8", "#0e8c6a", "#b8860b", "#c0392b", "#5b4fcf", "#0891b2"]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — PROBLEM TYPE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 0:
    section_header(1, "Problem Type", "Define the machine learning task for your healthcare finance analysis")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        choice = st.radio("", ["Classification", "Regression"],
                          horizontal=True, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            active_cls = "var(--accent)" if choice == "Classification" else "var(--border)"
            bg_cls = "var(--accent-light)" if choice == "Classification" else "var(--surface)"
            text_cls = "var(--accent)" if choice == "Classification" else "var(--text)"
            st.markdown(f"""
            <div style="background:{bg_cls};border:2px solid {active_cls};border-radius:10px;
                        padding:24px;text-align:center;">
                <div style="width:48px;height:48px;background:var(--accent);border-radius:8px;
                            display:flex;align-items:center;justify-content:center;
                            font-size:22px;margin:0 auto 12px;">🎯</div>
                <div style="font-weight:700;color:{text_cls};font-size:16px;margin-bottom:8px;">Classification</div>
                <div style="color:var(--muted);font-size:13px;line-height:1.5;">
                    Predict discrete outcomes such as high/low cost risk categories
                </div>
                <div style="margin-top:12px;font-size:11px;color:var(--accent);font-family:Inter,sans-serif;font-weight:600;letter-spacing:0.5px;">
                    LOGISTIC · SVM · RANDOM FOREST
                </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            active_reg = "var(--accent2)" if choice == "Regression" else "var(--border)"
            bg_reg = "var(--accent2-light)" if choice == "Regression" else "var(--surface)"
            text_reg = "var(--accent2)" if choice == "Regression" else "var(--text)"
            st.markdown(f"""
            <div style="background:{bg_reg};border:2px solid {active_reg};border-radius:10px;
                        padding:24px;text-align:center;">
                <div style="width:48px;height:48px;background:var(--accent2);border-radius:8px;
                            display:flex;align-items:center;justify-content:center;
                            font-size:22px;margin:0 auto 12px;">📈</div>
                <div style="font-weight:700;color:{text_reg};font-size:16px;margin-bottom:8px;">Regression</div>
                <div style="color:var(--muted);font-size:13px;line-height:1.5;">
                    Forecast continuous values like per-capita health expenditure
                </div>
                <div style="margin-top:12px;font-size:11px;color:var(--accent2);font-family:Inter,sans-serif;font-weight:600;letter-spacing:0.5px;">
                    LINEAR · SVR · RANDOM FOREST
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Continue →", use_container_width=True):
            st.session_state.problem_type = choice
            st.session_state.step = 1
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA INPUT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 1:
    section_header(2, "Data Input", "Upload your healthcare finance dataset and configure the prediction target")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px;margin-bottom:16px;
                    border-left:4px solid var(--accent);">
            <div style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:12px;">
                UPLOAD DATASET
            </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop your CSV / Excel file",
            type=["csv", "xlsx", "xls"],
            help="Healthcare Financing dataset (OWID) or any tabular CSV"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded:
            try:
                if uploaded.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded)
                else:
                    df = pd.read_csv(uploaded)

                st.session_state.df = df

                n_num = df.select_dtypes(include=np.number).shape[1]
                n_cat = df.select_dtypes(include='object').shape[1]
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

                st.markdown("""<div style="font-size:11px;color:var(--muted);text-transform:uppercase;
                               letter-spacing:1.5px;font-weight:600;margin:16px 0 8px;">TARGET VARIABLE</div>""",
                            unsafe_allow_html=True)
                target = st.selectbox("Select the column to predict", df.columns.tolist())
                st.session_state.target = target

                other_cols = [c for c in df.columns if c != target]
                st.markdown("""<div style="font-size:11px;color:var(--muted);text-transform:uppercase;
                               letter-spacing:1.5px;font-weight:600;margin:16px 0 8px;">INPUT FEATURES</div>""",
                            unsafe_allow_html=True)
                features = st.multiselect("Select features (default = all)", other_cols, default=other_cols)
                st.session_state.features = features

                if len(features) >= 2:
                    st.markdown("""<div style="font-size:11px;color:var(--muted);text-transform:uppercase;
                                   letter-spacing:1.5px;font-weight:600;margin:20px 0 8px;">PCA DATA SHAPE</div>""",
                                unsafe_allow_html=True)
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
                                             template="plotly_white",
                                             color_discrete_sequence=COLORS)
                        else:
                            fig = px.scatter_3d(x=comp[:,0], y=comp[:,1], z=comp[:,2],
                                                color=y_col,
                                                labels={"x":"PC1","y":"PC2","z":"PC3"},
                                                template="plotly_white",
                                                color_discrete_sequence=COLORS)
                            fig.update_traces(marker_size=3)

                        ev = pca.explained_variance_ratio_
                        title_ev = " | ".join([f"PC{i+1}: {v*100:.1f}%" for i, v in enumerate(ev)])
                        fig.update_layout(
                            height=400, margin=dict(l=0,r=0,t=40,b=0),
                            **PLOTLY_LAYOUT,
                            title=dict(text=f"PCA — {title_ev}", font=dict(color='#1a6fa8', size=13)),
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
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px;
                    border-left:4px solid var(--gold);">
            <div style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:16px;">
                DATASET GUIDE
            </div>
            <div style="font-size:13px;color:var(--text-body);line-height:1.8;">
                <div style="background:var(--accent-light);border:1px solid #b3d4ec;border-radius:6px;padding:12px;margin-bottom:12px;">
                    <b style="color:#0d3b5e;">📂 Financing Healthcare</b><br>
                    <span style="color:#2d5a7a;font-size:12px;">OWID dataset by Ortiz-Ospina &amp; Roser — global health expenditure indicators</span>
                </div>
                <div style="background:var(--gold-light);border:1px solid #d4a82a;border-radius:6px;padding:12px;margin-bottom:12px;">
                    <b style="color:#5a3c00;">🎯 Recommended targets</b><br>
                    <span style="color:#6b4a00;font-size:12px;">che_gdp · che_pc_usd · gghed_che</span>
                </div>
                <div style="background:var(--accent2-light);border:1px solid #6ec9a8;border-radius:6px;padding:12px;">
                    <b style="color:#055c3f;">💡 Tip</b><br>
                    <span style="color:#0a5c44;font-size:12px;">Drop identifier columns like 'country' before modeling to avoid data leakage.</span>
                </div>
                <hr style="border-color:var(--border);margin:16px 0;">
                <div style="font-size:12px;color:var(--muted);">
                    Supported formats: CSV · XLSX · XLS
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back", key="back1"):
            st.session_state.step = 0
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    section_header(3, "Exploratory Data Analysis", "Understand distributions, correlations and missing patterns in your healthcare data")

    df = st.session_state.df_clean
    target = st.session_state.target

    if df is None:
        st.warning("No data found. Go back to Step 2.")
    else:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()

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
                                template="plotly_white")
                fig.update_layout(height=600, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

                if target in numeric_cols:
                    top_corr = corr[target].drop(target).abs().sort_values(ascending=False).head(10)
                    fig2 = px.bar(x=top_corr.values, y=top_corr.index, orientation='h',
                                  color=top_corr.values,
                                  color_continuous_scale=['#c0392b','#b8860b','#1a6fa8'],
                                  template="plotly_white",
                                  labels={"x":"Absolute Correlation","y":""})
                    fig2.update_layout(height=350, **PLOTLY_LAYOUT,
                                       title=dict(text="Top correlations with target", font=dict(color='#1a2332')))
                    st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            miss = df.isnull().sum()
            miss = miss[miss > 0].sort_values(ascending=False)
            if miss.empty:
                st.success("✅ No missing values detected!")
            else:
                fig = px.bar(x=miss.index, y=miss.values,
                             color=miss.values, color_continuous_scale=["#fde8e6","#c0392b"],
                             template="plotly_white",
                             labels={"x":"Column","y":"Missing Count"})
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"**{len(miss)} columns** have missing values · **{df.isnull().sum().sum():,}** total NaN cells")

        with tab4:
            if target in numeric_cols:
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.histogram(df, x=target, template="plotly_white",
                                       color_discrete_sequence=["#1a6fa8"])
                    fig.update_layout(**PLOTLY_LAYOUT,
                                      title=dict(text=f"Distribution of {target}", font=dict(color='#1a2332')))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.box(df, y=target, template="plotly_white",
                                 color_discrete_sequence=["#0e8c6a"])
                    fig.update_layout(**PLOTLY_LAYOUT,
                                      title=dict(text=f"Box Plot — {target}", font=dict(color='#1a2332')))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(df[target].value_counts(), template="plotly_white",
                             color_discrete_sequence=["#1a6fa8"])
                fig.update_layout(**PLOTLY_LAYOUT,
                                  title=dict(text=f"Class Distribution — {target}", font=dict(color='#1a2332')))
                st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.step = 1; st.rerun()
        with col2:
            if st.button("Continue to Data Engineering →", use_container_width=True):
                st.session_state.step = 3; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DATA ENGINEERING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    section_header(4, "Data Engineering & Cleaning", "Handle missing values and detect/remove outliers")

    df = st.session_state.df_clean.copy()
    target = st.session_state.target
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("""<div style="font-size:11px;color:var(--muted);text-transform:uppercase;
                       letter-spacing:1.5px;font-weight:600;margin-bottom:8px;">MISSING VALUE STRATEGY</div>""",
                    unsafe_allow_html=True)

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

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("""<div style="font-size:11px;color:var(--muted);text-transform:uppercase;
                       letter-spacing:1.5px;font-weight:600;margin-bottom:8px;">OUTLIER DETECTION</div>""",
                    unsafe_allow_html=True)

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
                                     color_discrete_map={"Outlier":"#c0392b","Normal":"#1a6fa8"},
                                     template="plotly_white",
                                     title=f"Outliers via {outlier_method}")
                    fig.update_layout(**PLOTLY_LAYOUT, height=380,
                                      title=dict(text=f"Outliers via {outlier_method}", font=dict(color='#1a2332')))
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

    with col_r:
        st.markdown("""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px;
                    border-left:4px solid var(--accent2);">
            <div style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:14px;">
                CURRENT DATA STATE
            </div>
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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    section_header(5, "Feature Selection", "Identify the most informative healthcare finance predictors")

    df = st.session_state.df_clean.copy().dropna()
    target = st.session_state.target
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]

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
                    removed = [c for c, m in zip(numeric_cols, mask) if not m]

                    variances = df[numeric_cols].var().sort_values(ascending=False)
                    fig = px.bar(x=variances.index, y=variances.values,
                                 color=(variances >= thresh).map({True:"#1a6fa8", False:"#c0392b"}),
                                 template="plotly_white",
                                 labels={"x":"Feature","y":"Variance"})
                    fig.add_hline(y=thresh, line_dash="dash", line_color="#b8860b",
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
                             color=(corrs.sort_values(ascending=False) >= corr_thresh).map({True:"#1a6fa8",False:"#c0392b"}),
                             template="plotly_white",
                             labels={"x":"Feature","y":f"|Corr with {target}|"})
                fig.add_hline(y=corr_thresh, line_dash="dash", line_color="#b8860b")
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
                         color=(mi_series.index.isin(selected)).map({True:"#1a6fa8",False:"#0e8c6a"}),
                         template="plotly_white",
                         labels={"x":"Feature","y":"Mutual Information Score"})
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Top **{top_k}** features selected")

    with col_r:
        st.markdown("""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px;
                    border-left:4px solid var(--accent);">
            <div style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:14px;">
                SELECTED FEATURES
            </div>
        """, unsafe_allow_html=True)
        for feat in selected:
            st.markdown(f"""
            <div style="background:var(--accent-light);border:1px solid #b3d4ec;
                        border-radius:6px;padding:7px 12px;margin-bottom:6px;
                        font-size:13px;color:#0d3b5e;font-weight:500;">
                ✓ {feat}
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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — DATA SPLIT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    section_header(6, "Data Split", "Partition your healthcare dataset into training and testing sets")

    df = st.session_state.df_clean.dropna()
    target = st.session_state.target
    features = st.session_state.selected_features or [c for c in df.select_dtypes(include=np.number).columns if c != target]

    col_l, col_r = st.columns([2, 3])

    with col_l:
        test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
        random_state = st.number_input("Random seed", 0, 999, 42)

        # Only offer stratify option for classification
        stratify_requested = st.checkbox(
            "Stratify split (classification)",
            value=(st.session_state.problem_type == "Classification")
        )

        if st.button("✂️ Split Dataset", use_container_width=True):
            from sklearn.model_selection import train_test_split
            df_sub = df[features + [target]].dropna()
            X = df_sub[features]
            y = df_sub[target]

            # ── FIX: validate stratify feasibility before using it ──
            strat = None
            if stratify_requested and st.session_state.problem_type == "Classification":
                if can_stratify(y):
                    strat = y
                else:
                    st.warning(
                        "⚠️ **Stratification disabled automatically** — one or more classes have "
                        "only 1 sample, which is too few for stratified splitting. "
                        "Proceeding with a standard random split instead.\n\n"
                        "💡 Tip: If your target is a high-cardinality column like `country`, "
                        "consider using a numeric column (e.g. `che_gdp`) as the target instead."
                    )

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=strat
                )
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.test_size = test_size
                st.session_state.random_state = random_state
                st.success(
                    f"✅ Split successful! "
                    f"{'Stratified · ' if strat is not None else 'Random · '}"
                    f"Train: {len(X_train):,} · Test: {len(X_test):,}"
                )
            except Exception as e:
                st.error(f"Split error: {e}")

    with col_r:
        if st.session_state.X_train is not None:
            n_train = len(st.session_state.X_train)
            n_test = len(st.session_state.X_test)

            fig = go.Figure(data=[go.Pie(
                labels=["Train", "Test"],
                values=[n_train, n_test],
                marker=dict(colors=["#1a6fa8","#0e8c6a"]),
                hole=0.55,
                textinfo="label+percent",
                textfont_size=14
            )])
            fig.update_layout(height=320, **PLOTLY_LAYOUT, showlegend=False,
                              margin=dict(l=0,r=0,t=0,b=0),
                              annotations=[dict(text=f"{n_train+n_test:,}<br>total",
                                               x=0.5, y=0.5, showarrow=False,
                                               font=dict(size=16, color="#1a2332"))])
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Train samples", f"{n_train:,}")
            c2.metric("Test samples", f"{n_test:,}")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.step = 4; st.rerun()
    with col2:
        if st.session_state.X_train is not None:
            if st.button("Continue to Model Selection →", use_container_width=True):
                st.session_state.step = 6; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — MODEL SELECTION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 6:
    section_header(7, "Model Selection", "Choose the algorithm best suited for your healthcare finance prediction task")

    pt = st.session_state.problem_type

    if pt == "Classification":
        model_options = {
            "Logistic Regression": ("🔵", "Best for binary risk classification (e.g. high/low cost)"),
            "SVM (kernel options)": ("🔷", "Effective on high-dimensional healthcare feature spaces"),
            "Random Forest": ("🌲", "Robust ensemble method for complex expenditure patterns"),
            "K-Means (unsupervised)": ("⭕", "Cluster countries/regions by healthcare spending profile"),
        }
    else:
        model_options = {
            "Linear Regression": ("📏", "Baseline forecasting of healthcare expenditure per capita"),
            "SVR (kernel options)": ("🔷", "Non-linear health cost prediction with kernel trick"),
            "Random Forest Regressor": ("🌲", "Capture complex GDP-to-health-spend relationships"),
            "K-Means (unsupervised)": ("⭕", "Segment health systems by spending patterns"),
        }

    cols = st.columns(len(model_options))
    selected_model = st.session_state.model_name or list(model_options.keys())[0]

    for i, (name, (icon, desc)) in enumerate(model_options.items()):
        with cols[i]:
            active = selected_model == name
            border = "var(--accent)" if active else "var(--border)"
            bg = "var(--accent-light)" if active else "var(--surface)"
            desc_color = "#0d3b5e" if active else "var(--muted)"
            badge_text = f'<span style="color:var(--accent);font-weight:600;">✓ Selected</span>' if active else f'<span style="color:{desc_color};">{desc}</span>'

            if st.button(f"{icon} {name}", key=f"model_{i}", use_container_width=True):
                st.session_state.model_name = name
                st.rerun()
            st.markdown(f"""
            <div style="background:{bg};border:2px solid {border};border-radius:8px;
                        padding:10px;text-align:center;margin-top:-8px;font-size:12px;
                        line-height:1.5;">
                {badge_text}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    model_name = st.session_state.model_name
    if model_name and ("SVM" in model_name or "SVR" in model_name):
        st.markdown("""<div style="font-size:11px;color:var(--muted);text-transform:uppercase;
                       letter-spacing:1.5px;font-weight:600;margin-bottom:8px;">SVM KERNEL OPTIONS</div>""",
                    unsafe_allow_html=True)
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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — TRAINING & K-FOLD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 7:
    section_header(8, "Training & K-Fold Validation", "Train and validate your healthcare finance model with cross-validation")

    model_name = st.session_state.model_name
    pt = st.session_state.problem_type

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
                sc = StandardScaler()
                Xs = sc.fit_transform(X_train.fillna(0))
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

                        # ── FIX: use StratifiedKFold only when feasible ──
                        unique_counts = pd.Series(ys).value_counts()
                        if (unique_counts >= k).all():
                            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
                        else:
                            cv = KFold(n_splits=k, shuffle=True, random_state=42)
                            st.info(
                                f"ℹ️ Switched to standard KFold — some classes have fewer than {k} samples "
                                "so stratified splitting is not possible."
                            )
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
                                 marker_color=["#1a6fa8" if s >= mean_score else "#0e8c6a" for s in cv_scores],
                                 text=[f"{s:.4f}" for s in cv_scores],
                                 textposition="outside"))
            fig.add_hline(y=mean_score, line_dash="dash", line_color="#b8860b",
                          annotation_text=f"Mean = {mean_score:.4f}")
            fig.update_layout(template="plotly_white", height=380,
                              **PLOTLY_LAYOUT,
                              title=dict(text="Cross-Validation Scores per Fold", font=dict(color='#1a2332')),
                              yaxis_title="Score", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Mean", f"{cv_scores.mean():.4f}")
            c2.metric("Std Dev", f"{cv_scores.std():.4f}")
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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 8:
    section_header(9, "Performance Metrics", "Evaluate model quality and assess fit for healthcare finance predictions")

    model = st.session_state.trained_model
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    pt = st.session_state.problem_type
    le = st.session_state.get("label_encoder", None)

    if model is None or X_train is None:
        st.warning("Please train a model first.")
    else:
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train.fillna(0))
        X_test_s = sc.transform(X_test.fillna(0))

        y_train_raw = y_train.fillna(0)
        y_test_raw = y_test.fillna(0)

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
                fig = px.imshow(cm, text_auto=True, template="plotly_white",
                                color_continuous_scale=["#e8f0f7","#1a6fa8"],
                                labels=dict(x="Predicted", y="Actual"),
                                title="Confusion Matrix")
                fig.update_layout(**PLOTLY_LAYOUT, height=400,
                                  title=dict(text="Confusion Matrix", font=dict(color='#1a2332')))
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
                                 template="plotly_white",
                                 color_discrete_sequence=["#1a6fa8"],
                                 title="Actual vs Predicted")
                fig.add_shape(type="line", x0=float(y_test_enc.min()), y0=float(y_test_enc.min()),
                              x1=float(y_test_enc.max()), y1=float(y_test_enc.max()),
                              line=dict(color="#c0392b", dash="dash"))
                fig.update_layout(**PLOTLY_LAYOUT, height=400,
                                  title=dict(text="Actual vs Predicted Healthcare Expenditure", font=dict(color='#1a2332')))
                st.plotly_chart(fig, use_container_width=True)

                residuals = y_test_enc - test_preds
                fig2 = px.histogram(x=residuals, template="plotly_white",
                                    color_discrete_sequence=["#0e8c6a"],
                                    labels={"x":"Residual"},
                                    title="Residual Distribution")
                fig2.update_layout(**PLOTLY_LAYOUT, height=320,
                                   title=dict(text="Residual Distribution", font=dict(color='#1a2332')))
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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 9:
    section_header(10, "Hyperparameter Tuning", "Optimize your model for maximum predictive accuracy on healthcare finance data")

    model_name = st.session_state.model_name
    pt = st.session_state.problem_type
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    le = st.session_state.get("label_encoder", None)

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

        st.markdown("""<div style="font-size:11px;color:var(--muted);text-transform:uppercase;
                       letter-spacing:1.5px;font-weight:600;margin:16px 0 8px;">PARAMETER GRID</div>""",
                    unsafe_allow_html=True)
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
                    st.markdown("""<div style="font-size:11px;color:var(--muted);text-transform:uppercase;
                                   letter-spacing:1.5px;font-weight:600;margin:16px 0 8px;">BEST PARAMETERS</div>""",
                                unsafe_allow_html=True)
                    st.json(best_params)

                    results = pd.DataFrame(searcher.cv_results_).sort_values("rank_test_score")
                    show_cols = [c for c in results.columns if c.startswith("param_") or
                                 c in ["mean_test_score","std_test_score","rank_test_score"]]
                    st.markdown("""<div style="font-size:11px;color:var(--muted);text-transform:uppercase;
                                   letter-spacing:1.5px;font-weight:600;margin:16px 0 8px;">ALL RESULTS</div>""",
                                unsafe_allow_html=True)
                    st.dataframe(results[show_cols].head(20), use_container_width=True)

                    fig = px.bar(results.head(15), y="mean_test_score",
                                 error_y="std_test_score",
                                 color="mean_test_score",
                                 color_continuous_scale=["#c0392b","#b8860b","#1a6fa8"],
                                 template="plotly_white",
                                 title=f"Top 15 parameter combinations — {scoring}")
                    fig.update_layout(**PLOTLY_LAYOUT, height=400,
                                      xaxis_title="Combination #",
                                      yaxis_title=scoring, showlegend=False,
                                      title=dict(text=f"Top 15 parameter combinations — {scoring}", font=dict(color='#1a2332')))
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


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:32px 20px 20px;border-top:1px solid var(--border);margin-top:40px;">
    <div style="display:flex;justify-content:center;align-items:center;gap:24px;margin-bottom:12px;flex-wrap:wrap;">
        <span style="font-size:12px;color:var(--muted);">💊 Clinical Cost Modeling</span>
        <span style="color:var(--border);">|</span>
        <span style="font-size:12px;color:var(--muted);">📊 Expenditure Analytics</span>
        <span style="color:var(--border);">|</span>
        <span style="font-size:12px;color:var(--muted);">🌐 Global Health Finance</span>
    </div>
    <span style="font-family:Inter,sans-serif;font-size:11px;color:var(--muted);">
        ML Pipeline Studio · Healthcare Finance Analytics · Built with Streamlit + Plotly
    </span>
</div>
""", unsafe_allow_html=True)
