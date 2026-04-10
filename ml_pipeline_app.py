import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Pipeline · Healthcare Finance",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --border: #1f2d45;
    --accent: #00d4aa;
    --accent2: #7c6aff;
    --accent3: #ff6b6b;
    --text: #e2e8f0;
    --muted: #64748b;
    --card-bg: rgba(17,24,39,0.95);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 20% 20%, rgba(0,212,170,0.04) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(124,106,255,0.04) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stHeader"] { background: transparent !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #00b894) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,212,170,0.3) !important;
}

/* Selectbox, multiselect */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* Number input */
.stNumberInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* Slider */
.stSlider > div > div > div { background: var(--accent) !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 12px !important; }

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* Metric */
[data-testid="stMetric"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] { color: var(--accent) !important; font-size: 28px !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; }

/* Horizontal rule */
hr { border-color: var(--border) !important; }

/* Radio */
.stRadio > div { flex-direction: row !important; gap: 16px !important; }
.stRadio > div > label {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
.stRadio > div > label:hover { border-color: var(--accent) !important; }

/* Success / Warning / Error */
.stSuccess { background: rgba(0,212,170,0.1) !important; border: 1px solid var(--accent) !important; border-radius: 8px !important; }
.stWarning { background: rgba(255,193,7,0.1) !important; border-radius: 8px !important; }
.stError   { background: rgba(255,107,107,0.1) !important; border: 1px solid var(--accent3) !important; border-radius: 8px !important; }
.stInfo    { background: rgba(124,106,255,0.1) !important; border: 1px solid var(--accent2) !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def card(content_fn, title="", accent="var(--accent)"):
    st.markdown(f"""
    <div style="background:var(--card-bg);border:1px solid var(--border);border-radius:14px;
                padding:24px;margin-bottom:20px;border-top:3px solid {accent};">
        {'<p style="font-family:Space Mono,monospace;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:2px;margin-bottom:16px;">' + title + '</p>' if title else ''}
    """, unsafe_allow_html=True)
    content_fn()
    st.markdown("</div>", unsafe_allow_html=True)


def step_badge(n, label, done=False, active=False):
    if active:
        bg, col, border = "var(--accent)", "#0a0e1a", "var(--accent)"
    elif done:
        bg, col, border = "rgba(0,212,170,0.15)", "var(--accent)", "var(--accent)"
    else:
        bg, col, border = "var(--surface2)", "var(--muted)", "var(--border)"
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;gap:6px;min-width:90px;">
        <div style="width:40px;height:40px;border-radius:50%;background:{bg};border:2px solid {border};
                    display:flex;align-items:center;justify-content:center;
                    font-family:Space Mono,monospace;font-weight:700;font-size:13px;color:{col};">
            {'✓' if done else str(n)}
        </div>
        <span style="font-size:10px;color:{col};text-align:center;font-weight:500;line-height:1.3;">{label}</span>
    </div>"""


def connector(done=False):
    col = "var(--accent)" if done else "var(--border)"
    return f'<div style="flex:1;height:2px;background:{col};margin-top:-14px;"></div>'


STEPS = [
    "Problem\nType", "Data\nInput", "EDA", "Engineering\n& Cleaning",
    "Feature\nSelection", "Data\nSplit", "Model\nSelection",
    "Training &\nValidation", "Metrics", "Hyper-\nParameter Tuning"
]

def render_stepper(current_step):
    html = '<div style="display:flex;align-items:flex-start;gap:0;padding:20px 10px;overflow-x:auto;">'
    for i, label in enumerate(STEPS):
        done = i < current_step
        active = i == current_step
        html += step_badge(i + 1, label, done, active)
        if i < len(STEPS) - 1:
            html += connector(done)
    html += '</div>'
    st.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:16px;
                padding:12px 8px;margin-bottom:28px;">
        {html}
    </div>
    """, unsafe_allow_html=True)


def section_header(step_n, title, subtitle=""):
    st.markdown(f"""
    <div style="margin-bottom:24px;">
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;">
            <div style="width:36px;height:36px;border-radius:10px;
                        background:linear-gradient(135deg,var(--accent),#00b894);
                        display:flex;align-items:center;justify-content:center;
                        font-family:Space Mono,monospace;font-weight:700;font-size:13px;color:#0a0e1a;">
                {step_n:02d}
            </div>
            <h2 style="margin:0;font-size:22px;font-weight:700;color:var(--text);">{title}</h2>
        </div>
        {'<p style="color:var(--muted);margin:0 0 0 50px;font-size:14px;">' + subtitle + '</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


# ── State Initialization ──────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:40px 20px 20px;">
    <div style="font-family:Space Mono,monospace;font-size:11px;color:var(--accent);
                letter-spacing:4px;text-transform:uppercase;margin-bottom:10px;">
        AutoML · Healthcare Finance
    </div>
    <h1 style="font-size:42px;font-weight:700;margin:0;
               background:linear-gradient(135deg,#00d4aa,#7c6aff);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        ML Pipeline Studio
    </h1>
    <p style="color:var(--muted);font-size:15px;margin-top:10px;">
        End-to-end machine learning · from raw data to tuned model
    </p>
</div>
""", unsafe_allow_html=True)

render_stepper(st.session_state.step)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — PROBLEM TYPE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 0:
    section_header(1, "Problem Type", "What kind of ML task are you solving?")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        choice = st.radio("", ["Classification", "Regression"],
                          horizontal=True, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)

        # Visual cards
        c1, c2 = st.columns(2)
        with c1:
            active_cls = "#00d4aa" if choice == "Classification" else "var(--border)"
            st.markdown(f"""
            <div style="background:var(--surface2);border:2px solid {active_cls};border-radius:14px;
                        padding:20px;text-align:center;transition:all 0.3s;">
                <div style="font-size:32px;margin-bottom:10px;">🎯</div>
                <div style="font-weight:700;color:var(--text);font-size:16px;">Classification</div>
                <div style="color:var(--muted);font-size:13px;margin-top:8px;">
                    Predict discrete labels or categories
                </div>
                <div style="margin-top:12px;font-size:11px;color:var(--accent);font-family:Space Mono,monospace;">
                    Logistic · SVM · Random Forest
                </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            active_reg = "#7c6aff" if choice == "Regression" else "var(--border)"
            st.markdown(f"""
            <div style="background:var(--surface2);border:2px solid {active_reg};border-radius:14px;
                        padding:20px;text-align:center;transition:all 0.3s;">
                <div style="font-size:32px;margin-bottom:10px;">📈</div>
                <div style="font-weight:700;color:var(--text);font-size:16px;">Regression</div>
                <div style="color:var(--muted);font-size:13px;margin-top:8px;">
                    Predict continuous numerical values
                </div>
                <div style="margin-top:12px;font-size:11px;color:var(--accent2);font-family:Space Mono,monospace;">
                    Linear · SVR · Random Forest
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
    section_header(2, "Data Input", "Upload your dataset and configure the target variable")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:16px;">
            <div style="font-family:Space Mono,monospace;font-size:11px;color:var(--accent);letter-spacing:2px;margin-bottom:12px;">
                UPLOAD DATASET
            </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop your CSV / Excel file",
            type=["csv", "xlsx", "xls"],
            help="Financing Healthcare dataset or any tabular CSV"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded:
            try:
                if uploaded.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded)
                else:
                    df = pd.read_csv(uploaded)

                st.session_state.df = df

                # Dataset stats
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

                # Target selection
                st.markdown("""
                <div style="font-family:Space Mono,monospace;font-size:11px;color:var(--accent);
                            letter-spacing:2px;margin:16px 0 8px;">TARGET FEATURE</div>""",
                            unsafe_allow_html=True)
                target = st.selectbox("Select the column to predict", df.columns.tolist())
                st.session_state.target = target

                # Feature selection
                other_cols = [c for c in df.columns if c != target]
                st.markdown("""
                <div style="font-family:Space Mono,monospace;font-size:11px;color:var(--accent);
                            letter-spacing:2px;margin:16px 0 8px;">INPUT FEATURES</div>""",
                            unsafe_allow_html=True)
                features = st.multiselect("Select features (default = all)", other_cols, default=other_cols)
                st.session_state.features = features

                # PCA 2D/3D scatter
                if len(features) >= 2:
                    st.markdown("""
                    <div style="font-family:Space Mono,monospace;font-size:11px;color:var(--accent2);
                                letter-spacing:2px;margin:20px 0 8px;">PCA DATA SHAPE</div>""",
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
                                             template="plotly_dark",
                                             color_discrete_sequence=px.colors.qualitative.Bold)
                        else:
                            fig = px.scatter_3d(x=comp[:,0], y=comp[:,1], z=comp[:,2],
                                                color=y_col,
                                                labels={"x":"PC1","y":"PC2","z":"PC3"},
                                                template="plotly_dark",
                                                color_discrete_sequence=px.colors.qualitative.Bold)
                            fig.update_traces(marker_size=3)

                        ev = pca.explained_variance_ratio_
                        title_ev = " | ".join([f"PC{i+1}: {v*100:.1f}%" for i, v in enumerate(ev)])
                        fig.update_layout(
                            height=400, margin=dict(l=0,r=0,t=40,b=0),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            title=dict(text=f"PCA — {title_ev}", font=dict(color='#00d4aa', size=13)),
                            legend=dict(bgcolor='rgba(0,0,0,0)')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Need ≥2 numeric features for PCA")

                if st.button("Continue to EDA →", use_container_width=True):
                    st.session_state.df_clean = df[features + [target]].copy()
                    st.session_state.step = 2
                    st.rerun()

            except Exception as e:
                st.error(f"Error loading file: {e}")

    with col_right:
        st.markdown("""
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:20px;">
            <div style="font-family:Space Mono,monospace;font-size:11px;color:var(--accent2);
                        letter-spacing:2px;margin-bottom:16px;">DATASET GUIDE</div>
            <div style="font-size:13px;color:var(--muted);line-height:1.8;">
                <p>📂 <b style="color:var(--text);">Financing Healthcare</b><br>
                   OWID dataset by Ortiz-Ospina & Roser.</p>
                <p>🎯 <b style="color:var(--text);">Recommended targets</b><br>
                   che_gdp, che_pc_usd, gghed_che</p>
                <p>💡 <b style="color:var(--text);">Tip</b><br>
                   Drop identifier columns like 'country' before modeling.</p>
                <hr style="border-color:var(--border);margin:16px 0;">
                <div style="font-size:12px;color:var(--muted);">
                    Supported formats: CSV · XLSX · XLS
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.df is None:
            st.markdown("<br>", unsafe_allow_html=True)
            # back button
        if st.button("← Back", key="back1"):
            st.session_state.step = 0
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    section_header(3, "Exploratory Data Analysis", "Understand distributions, correlations and missing patterns")

    df = st.session_state.df_clean
    target = st.session_state.target

    if df is None:
        st.warning("No data found. Go back to Step 2.")
    else:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()

        # Summary stats
        with st.expander("📊 Descriptive Statistics", expanded=True):
            st.dataframe(df.describe().T, use_container_width=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📉 Distributions", "🔗 Correlation", "❓ Missing Data", "🎯 Target Analysis"])

        with tab1:
            cols_to_plot = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:6])
            if cols_to_plot:
                n_cols = 3
                n_rows = -(-len(cols_to_plot) // n_cols)
                fig = make_subplots(rows=n_rows, cols=n_cols,
                                    subplot_titles=cols_to_plot)
                for idx, col in enumerate(cols_to_plot):
                    r, c = divmod(idx, n_cols)
                    fig.add_trace(go.Histogram(x=df[col].dropna(), name=col,
                                               marker_color=f'#{["00d4aa","7c6aff","ff6b6b","ffd166","06d6a0","ef476f"][idx%6]}',
                                               showlegend=False), row=r+1, col=c+1)
                fig.update_layout(height=300*n_rows, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                template="plotly_dark")
                fig.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)',
                                  font_color='#e2e8f0')
                st.plotly_chart(fig, use_container_width=True)

                # Top correlations with target
                if target in numeric_cols:
                    top_corr = corr[target].drop(target).abs().sort_values(ascending=False).head(10)
                    fig2 = px.bar(x=top_corr.values, y=top_corr.index, orientation='h',
                                  color=top_corr.values,
                                  color_continuous_scale=['#ff6b6b','#ffd166','#00d4aa'],
                                  template="plotly_dark",
                                  labels={"x":"Absolute Correlation","y":""})
                    fig2.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)',
                                       font_color='#e2e8f0', title="Top correlations with target")
                    st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            miss = df.isnull().sum()
            miss = miss[miss > 0].sort_values(ascending=False)
            if miss.empty:
                st.success("✅ No missing values detected!")
            else:
                fig = px.bar(x=miss.index, y=miss.values,
                             color=miss.values, color_continuous_scale="Reds",
                             template="plotly_dark",
                             labels={"x":"Column","y":"Missing Count"})
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"**{len(miss)} columns** have missing values · **{df.isnull().sum().sum():,}** total NaN cells")

        with tab4:
            if target in numeric_cols:
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.histogram(df, x=target, template="plotly_dark",
                                       color_discrete_sequence=["#00d4aa"])
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                                      title=f"Distribution of {target}")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.box(df, y=target, template="plotly_dark",
                                 color_discrete_sequence=["#7c6aff"])
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                                      title=f"Box Plot — {target}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(df[target].value_counts(), template="plotly_dark",
                             color_discrete_sequence=["#00d4aa"])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                                  title=f"Class Distribution — {target}")
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
        # ── Missing Value Imputation ──
        st.markdown("""<div style="font-family:Space Mono,monospace;font-size:11px;color:var(--accent);
                       letter-spacing:2px;margin-bottom:8px;">MISSING VALUE STRATEGY</div>""",
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

        # ── Outlier Detection ──
        st.markdown("""<div style="font-family:Space Mono,monospace;font-size:11px;color:var(--accent);
                       letter-spacing:2px;margin-bottom:8px;">OUTLIER DETECTION</div>""",
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

                # Viz
                if len(feat_for_outlier) >= 2:
                    is_out = pd.Series(df.index.isin(outlier_idx), index=df.index)
                    color_labels = is_out.map({True:"Outlier", False:"Normal"})
                    fig = px.scatter(df, x=feat_for_outlier[0], y=feat_for_outlier[1],
                                     color=color_labels,
                                     color_discrete_map={"Outlier":"#ff6b6b","Normal":"#00d4aa"},
                                     template="plotly_dark",
                                     title=f"Outliers via {outlier_method}")
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                                      height=380)
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
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:20px;">
            <div style="font-family:Space Mono,monospace;font-size:11px;color:var(--accent2);
                        letter-spacing:2px;margin-bottom:14px;">CURRENT DATA STATE</div>
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
    section_header(5, "Feature Selection", "Select the most informative features for your model")

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
                                 color=(variances >= thresh).map({True:"#00d4aa", False:"#ff6b6b"}),
                                 template="plotly_dark",
                                 labels={"x":"Feature","y":"Variance"})
                    fig.add_hline(y=thresh, line_dash="dash", line_color="#ffd166",
                                  annotation_text=f"Threshold={thresh}")
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                                      showlegend=False, height=380)
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
                             color=(corrs.sort_values(ascending=False) >= corr_thresh).map({True:"#00d4aa",False:"#ff6b6b"}),
                             template="plotly_dark",
                             labels={"x":"Feature","y":f"|Corr with {target}|"})
                fig.add_hline(y=corr_thresh, line_dash="dash", line_color="#ffd166")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                                  showlegend=False, height=380)
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
                         color=(mi_series.index.isin(selected)).map({True:"#00d4aa",False:"#7c6aff"}),
                         template="plotly_dark",
                         labels={"x":"Feature","y":"Mutual Information Score"})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                              showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Top **{top_k}** features selected")

    with col_r:
        st.markdown("""
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:20px;">
            <div style="font-family:Space Mono,monospace;font-size:11px;color:var(--accent);
                        letter-spacing:2px;margin-bottom:14px;">SELECTED FEATURES</div>
        """, unsafe_allow_html=True)
        for feat in selected:
            st.markdown(f"""
            <div style="background:rgba(0,212,170,0.08);border:1px solid rgba(0,212,170,0.2);
                        border-radius:6px;padding:6px 10px;margin-bottom:6px;
                        font-size:13px;color:var(--text);">
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
    section_header(6, "Data Split", "Partition data into training and testing sets")

    df = st.session_state.df_clean.dropna()
    target = st.session_state.target
    features = st.session_state.selected_features or [c for c in df.select_dtypes(include=np.number).columns if c != target]

    col_l, col_r = st.columns([2, 3])

    with col_l:
        test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
        random_state = st.number_input("Random seed", 0, 999, 42)
        stratify = st.checkbox("Stratify split (classification)", value=(st.session_state.problem_type=="Classification"))

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
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.test_size = test_size
                st.session_state.random_state = random_state
                st.success("✅ Split successful!")
            except Exception as e:
                st.error(f"Split error: {e}")

    with col_r:
        if st.session_state.X_train is not None:
            n_train = len(st.session_state.X_train)
            n_test = len(st.session_state.X_test)

            fig = go.Figure(data=[go.Pie(
                labels=["Train", "Test"],
                values=[n_train, n_test],
                marker=dict(colors=["#00d4aa","#7c6aff"]),
                hole=0.55,
                textinfo="label+percent",
                textfont_size=14
            )])
            fig.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)',
                              font_color='#e2e8f0', showlegend=False,
                              margin=dict(l=0,r=0,t=0,b=0),
                              annotations=[dict(text=f"{n_train+n_test:,}<br>total",
                                               x=0.5, y=0.5, showarrow=False,
                                               font=dict(size=16, color="#e2e8f0"))])
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
    section_header(7, "Model Selection", "Choose the algorithm that best fits your problem")

    pt = st.session_state.problem_type

    if pt == "Classification":
        model_options = {
            "Logistic Regression": "🔵",
            "SVM (kernel options)": "🔷",
            "Random Forest": "🌲",
            "K-Means (unsupervised)": "⭕"
        }
    else:
        model_options = {
            "Linear Regression": "📏",
            "SVR (kernel options)": "🔷",
            "Random Forest Regressor": "🌲",
            "K-Means (unsupervised)": "⭕"
        }

    cols = st.columns(len(model_options))
    selected_model = st.session_state.model_name or list(model_options.keys())[0]

    for i, (name, icon) in enumerate(model_options.items()):
        with cols[i]:
            active = selected_model == name
            border = "var(--accent)" if active else "var(--border)"
            bg = "rgba(0,212,170,0.08)" if active else "var(--surface2)"
            if st.button(f"{icon}\n{name}", key=f"model_{i}", use_container_width=True):
                st.session_state.model_name = name
                st.rerun()
            st.markdown(f"""
            <div style="background:{bg};border:2px solid {border};border-radius:10px;
                        padding:10px;text-align:center;margin-top:-10px;font-size:12px;color:var(--muted);">
                {'✓ Selected' if active else 'Click to select'}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Extra options for SVM
    model_name = st.session_state.model_name
    if model_name and "SVM" in model_name or (model_name and "SVR" in model_name):
        st.markdown("""<div style="font-family:Space Mono,monospace;font-size:11px;color:var(--accent);
                       letter-spacing:2px;margin-bottom:8px;">SVM KERNEL OPTIONS</div>""",
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
    section_header(8, "Training & K-Fold Validation", "Train your model with cross-validation")

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
                Xs = StandardScaler().fit_transform(X_train.fillna(0))
                ys = y_train.fillna(0)

                # Encode target if classification
                le = None
                if pt == "Classification":
                    if ys.dtype == object:
                        le = LabelEncoder()
                        ys = le.fit_transform(ys)
                    else:
                        ys = ys.astype(int)

                # Build model
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
                        else:  # KMeans
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

                    else:  # Regression
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

            fig = go.Figure()
            fig.add_trace(go.Bar(x=folds, y=cv_scores,
                                 marker_color=["#00d4aa" if s >= np.mean(cv_scores) else "#7c6aff" for s in cv_scores],
                                 text=[f"{s:.4f}" for s in cv_scores],
                                 textposition="outside"))
            fig.add_hline(y=np.mean(cv_scores), line_dash="dash", line_color="#ffd166",
                          annotation_text=f"Mean = {np.mean(cv_scores):.4f}")
            fig.update_layout(template="plotly_dark", height=380,
                              paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                              title="Cross-Validation Scores per Fold",
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
    section_header(9, "Performance Metrics", "Evaluate model quality and detect overfitting / underfitting")

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
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train.fillna(0))
        X_test_s = sc.transform(X_test.fillna(0))

        y_train_raw = y_train.fillna(0)
        y_test_raw = y_test.fillna(0)

        if pt == "Classification" and le is not None:
            y_train_enc = le.transform(y_train_raw.astype(str)) if hasattr(le,'transform') else y_train_raw.astype(int)
            y_test_enc  = le.transform(y_test_raw.astype(str))  if hasattr(le,'transform') else y_test_raw.astype(int)
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
                from sklearn.metrics import (accuracy_score, classification_report,
                                             confusion_matrix, roc_auc_score)
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

                # Confusion Matrix
                cm = confusion_matrix(y_test_enc, test_preds)
                fig = px.imshow(cm, text_auto=True, template="plotly_dark",
                                color_continuous_scale="Teal",
                                labels=dict(x="Predicted", y="Actual"),
                                title="Confusion Matrix")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=400)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📋 Classification Report"):
                    report = classification_report(y_test_enc, test_preds, output_dict=True)
                    st.dataframe(pd.DataFrame(report).T, use_container_width=True)

            else:  # Regression
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

                # Actual vs Predicted
                fig = px.scatter(x=y_test_enc, y=test_preds,
                                 labels={"x":"Actual","y":"Predicted"},
                                 template="plotly_dark",
                                 color_discrete_sequence=["#00d4aa"],
                                 title="Actual vs Predicted")
                fig.add_shape(type="line", x0=float(y_test_enc.min()), y0=float(y_test_enc.min()),
                              x1=float(y_test_enc.max()), y1=float(y_test_enc.max()),
                              line=dict(color="#ff6b6b", dash="dash"))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Residuals
                residuals = y_test_enc - test_preds
                fig2 = px.histogram(x=residuals, template="plotly_dark",
                                    color_discrete_sequence=["#7c6aff"],
                                    labels={"x":"Residual"},
                                    title="Residual Distribution")
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=320)
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
    section_header(10, "Hyperparameter Tuning", "Optimize your model with GridSearch or RandomizedSearch")

    model_name = st.session_state.model_name
    pt = st.session_state.problem_type
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    le = st.session_state.get("label_encoder", None)

    if X_train is None:
        st.warning("Please complete training first.")
    else:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train.fillna(0))
        y_train_raw = y_train.fillna(0)
        if pt == "Classification" and le is not None:
            try:
                y_enc = le.transform(y_train_raw.astype(str))
            except:
                y_enc = y_train_raw.astype(int)
        else:
            y_enc = y_train_raw

        search_method = st.radio("Search method", ["GridSearchCV", "RandomizedSearchCV"], horizontal=True)
        cv_k = st.number_input("CV folds for tuning", 2, 10, 3)

        # ── Model-specific param grids ──
        param_grids = {}

        if "Logistic" in (model_name or "") or "Linear Regression" in (model_name or ""):
            if "Logistic" in (model_name or ""):
                param_grids = {
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "solver": ["lbfgs", "liblinear"],
                    "max_iter": [100, 300, 500]
                }
            else:
                param_grids = {"fit_intercept": [True, False], "copy_X": [True]}

        elif "SVM" in (model_name or "") or "SVR" in (model_name or ""):
            param_grids = {
                "C": [0.01, 0.1, 1, 10],
                "kernel": ["rbf", "linear", "poly"],
                "gamma": ["scale", "auto"]
            }

        elif "Random Forest" in (model_name or ""):
            param_grids = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
        elif "K-Means" in (model_name or ""):
            param_grids = {"n_clusters": [2, 3, 4, 5, 6, 7, 8]}

        # Display param grid UI
        st.markdown("""<div style="font-family:Space Mono,monospace;font-size:11px;color:var(--accent);
                       letter-spacing:2px;margin:16px 0 8px;">PARAMETER GRID</div>""",
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

                # Rebuild base model
                try:
                    if "Logistic" in model_name:
                        from sklearn.linear_model import LogisticRegression
                        base = LogisticRegression(random_state=42)
                        scoring = "accuracy"
                    elif "SVM" in model_name:
                        from sklearn.svm import SVC
                        base = SVC(random_state=42)
                        scoring = "accuracy"
                    elif "Random Forest" in model_name and pt == "Classification":
                        from sklearn.ensemble import RandomForestClassifier
                        base = RandomForestClassifier(random_state=42)
                        scoring = "accuracy"
                    elif "Linear Regression" in model_name:
                        from sklearn.linear_model import LinearRegression
                        base = LinearRegression()
                        scoring = "r2"
                    elif "SVR" in model_name:
                        from sklearn.svm import SVR
                        base = SVR()
                        scoring = "r2"
                    elif "Random Forest" in model_name and pt == "Regression":
                        from sklearn.ensemble import RandomForestRegressor
                        base = RandomForestRegressor(random_state=42)
                        scoring = "r2"
                    else:
                        st.warning("Hyperparameter tuning not supported for K-Means via CV.")
                        st.stop()

                    with st.spinner("Running search... this may take a moment ⏳"):
                        if search_method == "GridSearchCV":
                            searcher = GridSearchCV(base, param_grids, cv=cv_k,
                                                    scoring=scoring, n_jobs=-1, refit=True)
                        else:
                            searcher = RandomizedSearchCV(base, param_grids, n_iter=n_iter,
                                                          cv=cv_k, scoring=scoring,
                                                          n_jobs=-1, refit=True, random_state=42)
                        searcher.fit(X_train_s, y_enc)

                    best_params = searcher.best_params_
                    best_score  = searcher.best_score_

                    st.success(f"✅ Best {scoring}: **{best_score:.4f}**")
                    st.markdown("""<div style="font-family:Space Mono,monospace;font-size:11px;color:var(--accent);
                                   letter-spacing:2px;margin:16px 0 8px;">BEST PARAMETERS</div>""",
                                unsafe_allow_html=True)
                    st.json(best_params)

                    # Results table
                    results = pd.DataFrame(searcher.cv_results_).sort_values("rank_test_score")
                    show_cols = [c for c in results.columns if c.startswith("param_") or
                                 c in ["mean_test_score","std_test_score","rank_test_score"]]
                    st.markdown("""<div style="font-family:Space Mono,monospace;font-size:11px;color:var(--accent);
                                   letter-spacing:2px;margin:16px 0 8px;">ALL RESULTS</div>""",
                                unsafe_allow_html=True)
                    st.dataframe(results[show_cols].head(20), use_container_width=True)

                    # Performance chart
                    fig = px.bar(results.head(15), y="mean_test_score",
                                 error_y="std_test_score",
                                 color="mean_test_score",
                                 color_continuous_scale=["#ff6b6b","#ffd166","#00d4aa"],
                                 template="plotly_dark",
                                 title=f"Top 15 parameter combinations — {scoring}")
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                                      height=400, xaxis_title="Combination #",
                                      yaxis_title=scoring, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # Update model with best
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


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:40px 20px 20px;border-top:1px solid var(--border);margin-top:40px;">
    <span style="font-family:Space Mono,monospace;font-size:11px;color:var(--muted);">
        ML Pipeline Studio · Financing Healthcare · Built with Streamlit + Plotly
    </span>
</div>
""", unsafe_allow_html=True)
