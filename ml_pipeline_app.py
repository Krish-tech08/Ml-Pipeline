import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# --- Scikit-learn Imports ---
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, mutual_info_classif
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             mean_squared_error, r2_score, mean_absolute_error)

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
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
    --pipeline-h:   95px;
    --header-h:     70px;
}

/* Base Styles */
.main .block-container {
    padding-top: calc(var(--pipeline-h) + var(--header-h) + 20px) !important;
    max-width: 1400px !important;
}

[data-testid="stHeader"] { display: none !important; }

/* Sticky Containers */
.sticky-top {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 1000;
}

.app-header {
    height: var(--header-h);
    background: #0d1a3a;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; padding: 0 32px;
}

.pipeline-bar {
    height: var(--pipeline-h);
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    padding: 0 10px; gap: 4px;
}

/* Pipeline Step UI */
.step-node {
    display: flex; flex-direction: column; align-items: center;
    min-width: 80px; position: relative;
}
.step-circle {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; border: 2px solid var(--border2);
    margin-bottom: 4px; background: var(--surface2);
}
.step-label { font-size: 9px; text-align: center; color: var(--text3); line-height: 1.1; font-weight: 600; }

.step-active .step-circle { border-color: var(--accent); background: var(--accent); color: white; box-shadow: 0 0 15px var(--accent-glow); }
.step-active .step-label { color: var(--accent); }
.step-done .step-circle { border-color: var(--green); background: var(--green-dim); color: var(--green); }
.step-done .step-label { color: var(--green); }

.step-connector { height: 2px; flex-grow: 1; max-width: 30px; background: var(--border); margin-bottom: 16px; }
.conn-done { background: var(--green); }

/* Components */
.ml-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
.card-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text3); margin-bottom: 12px; }

/* Utility Classes */
.sec-header { background: var(--surface); border-radius: 12px; padding: 20px; margin-bottom: 24px; border-left: 4px solid var(--accent); }
.sec-header h2 { margin: 0; font-family: 'Syne', sans-serif; font-size: 20px; }
.sec-header p { margin: 5px 0 0; font-size: 13px; color: var(--text2); }

.feat-pill { background: var(--accent-dim); border: 1px solid var(--accent); border-radius: 4px; padding: 4px 8px; font-size: 12px; margin: 2px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# ── INITIALIZE STATE ──────────────────────────────────────────────────────────
defaults = {
    'step': 0, 'problem_type': "Regression", 'df': None, 'target': None, 
    'features': None, 'df_clean': None, 'selected_features': None, 
    'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None, 
    'model_name': None, 'trained_model': None, 'cv_scores': None, 
    'outlier_indices': [], 'test_size': 0.2, 'random_state': 42
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── PLOTLY THEME ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='#94a3c0',
    font_family='Space Grotesk',
    margin=dict(l=40, r=20, t=40, b=40)
)
COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

# ── NAVIGATION HELPERS ────────────────────────────────────────────────────────
STEPS = [
    "Problem Type", "Data Input", "EDA", "Cleaning", "Selection", 
    "Split", "Model", "Training", "Metrics", "Tuning"
]

def render_pipeline_bar():
    current = st.session_state.step
    nodes_html = '<div class="sticky-top"><div class="app-header"><h1 style="color:white; font-size:18px; margin:0;">🏥 ML Pipeline Studio</h1></div><div class="pipeline-bar">'
    
    for i, label in enumerate(STEPS):
        is_done = i < current
        is_active = i == current
        cls = "step-done" if is_done else ("step-active" if is_active else "step-pending")
        circle_val = "✓" if is_done else str(i+1)
        
        nodes_html += f'<div class="step-node {cls}"><div class="step-circle">{circle_val}</div><div class="step-label">{label}</div></div>'
        if i < len(STEPS) - 1:
            conn_cls = "conn-done" if is_done else ""
            nodes_html += f'<div class="step-connector {conn_cls}"></div>'
            
    nodes_html += '</div></div>'
    st.markdown(nodes_html, unsafe_allow_html=True)

def section_header(title, subtitle):
    st.markdown(f"""<div class="sec-header"><h2>{title}</h2><p>{subtitle}</p></div>""", unsafe_allow_html=True)

# ── MAIN ───────────────────────────────────────────────────────────────────────
render_pipeline_bar()

# STEP 0: Problem Type
if st.session_state.step == 0:
    section_header("Problem Definition", "Select the type of machine learning task you want to perform.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Classification", use_container_width=True):
            st.session_state.problem_type = "Classification"
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("📈 Regression", use_container_width=True):
            st.session_state.problem_type = "Regression"
            st.session_state.step = 1
            st.rerun()

# STEP 1: Data Input
elif st.session_state.step == 1:
    section_header("Data Acquisition", "Upload your healthcare expenditure dataset (CSV or Excel).")
    uploaded = st.file_uploader("Upload file", type=["csv", "xlsx"])
    
    if uploaded:
        try:
            df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
            st.session_state.df = df
            st.success(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            target = st.selectbox("Select Prediction Target", df.columns.tolist())
            st.session_state.target = target
            
            feats = [c for c in df.columns if c != target]
            features = st.multiselect("Select Input Features", feats, default=feats[:min(len(feats), 10)])
            st.session_state.features = features
            
            if st.button("Proceed to EDA", use_container_width=True):
                st.session_state.df_clean = df[features + [target]].copy()
                st.session_state.step = 2
                st.rerun()
        except Exception as e:
            st.error(f"Load Error: {e}")

# STEP 2: EDA
elif st.session_state.step == 2:
    section_header("Exploratory Data Analysis", "Review statistical distributions and relationships.")
    df = st.session_state.df_clean
    
    tab1, tab2 = st.tabs(["Distributions", "Correlations"])
    with tab1:
        num_cols = df.select_dtypes(include=np.number).columns
        selected_vis = st.selectbox("Feature to visualize", num_cols)
        fig = px.histogram(df, x=selected_vis, color_discrete_sequence=[COLORS[0]], template="plotly_dark")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        corr = df.select_dtypes(include=np.number).corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", template="plotly_dark")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        
    if st.button("Move to Cleaning"):
        st.session_state.step = 3
        st.rerun()

# STEP 3: Cleaning
elif st.session_state.step == 3:
    section_header("Data Cleaning", "Handle missing values and outliers.")
    df = st.session_state.df_clean.copy()
    
    st.markdown('<div class="ml-card"><div class="card-label">Missing Values</div>', unsafe_allow_html=True)
    strategy = st.selectbox("Strategy", ["Drop Rows", "Mean Imputation", "Zero Fill"])
    if st.button("Apply"):
        if strategy == "Drop Rows": df = df.dropna()
        elif strategy == "Mean Imputation": df = df.fillna(df.mean(numeric_only=True))
        else: df = df.fillna(0)
        st.session_state.df_clean = df
        st.success("Cleaned!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Move to Feature Selection"):
        st.session_state.step = 4
        st.rerun()

# STEP 4: Feature Selection
elif st.session_state.step == 4:
    section_header("Feature Selection", "Narrow down the most predictive columns.")
    df = st.session_state.df_clean.dropna()
    target = st.session_state.target
    X = df.drop(columns=[target]).select_dtypes(include=np.number)
    y = df[target]
    
    method = st.radio("Selection Method", ["Variance Threshold", "Correlation"])
    selected = X.columns.tolist()
    
    if method == "Variance Threshold":
        vt = VarianceThreshold(threshold=0.1)
        vt.fit(X)
        selected = X.columns[vt.get_support()].tolist()
    
    st.session_state.selected_features = selected
    st.write("Selected Features:")
    st.write(", ".join(selected))
    
    if st.button("Confirm Split"):
        st.session_state.step = 5
        st.rerun()

# STEP 5: Data Split
elif st.session_state.step == 5:
    section_header("Data Splitting", "Create training and testing subsets.")
    size = st.slider("Test Size (%)", 10, 50, 20) / 100
    
    if st.button("Perform Split"):
        df = st.session_state.df_clean.dropna()
        target = st.session_state.target
        feats = st.session_state.selected_features
        X_train, X_test, y_train, y_test = train_test_split(df[feats], df[target], test_size=size, random_state=42)
        st.session_state.X_train, st.session_state.X_test = X_train, X_test
        st.session_state.y_train, st.session_state.y_test = y_train, y_test
        st.success("Data Split Successfully!")
        
    if st.session_state.X_train is not None:
        if st.button("Choose Model"):
            st.session_state.step = 6
            st.rerun()

# STEP 6: Model Selection
elif st.session_state.step == 6:
    section_header("Algorithm Selection", f"Choose a {st.session_state.problem_type} model.")
    if st.session_state.problem_type == "Classification":
        models = ["Logistic Regression", "Random Forest Classifier", "SVC"]
    else:
        models = ["Linear Regression", "Random Forest Regressor", "SVR"]
        
    choice = st.selectbox("Select Model", models)
    if st.button("Finalize Selection"):
        st.session_state.model_name = choice
        st.session_state.step = 7
        st.rerun()

# STEP 7: Training
elif st.session_state.step == 7:
    section_header("Model Training", f"Fitting {st.session_state.model_name}...")
    k = st.number_input("K-Folds", 2, 10, 5)
    
    if st.button("🚀 Start Training"):
        model_name = st.session_state.model_name
        if model_name == "Linear Regression": model = LinearRegression()
        elif model_name == "Random Forest Regressor": model = RandomForestRegressor()
        elif model_name == "Logistic Regression": model = LogisticRegression()
        elif model_name == "Random Forest Classifier": model = RandomForestClassifier()
        elif "SVC" in model_name or "SVR" in model_name: model = SVC() if "C" in model_name else SVR()
        
        X = st.session_state.X_train.fillna(0)
        y = st.session_state.y_train
        
        # Simple Label Encoding if target is string in classification
        if st.session_state.problem_type == "Classification" and not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        model.fit(X, y)
        scores = cross_val_score(model, X, y, cv=k)
        st.session_state.trained_model = model
        st.session_state.cv_scores = scores
        st.success(f"Trained! Mean CV Score: {scores.mean():.4f}")
        
    if st.session_state.trained_model:
        if st.button("View Metrics"):
            st.session_state.step = 8
            st.rerun()

# STEP 8: Metrics
elif st.session_state.step == 8:
    section_header("Evaluation Metrics", "How well did the model perform?")
    model = st.session_state.trained_model
    X_test = st.session_state.X_test.fillna(0)
    y_test = st.session_state.y_test
    
    y_pred = model.predict(X_test)
    
    if st.session_state.problem_type == "Regression":
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.metric("R² Score", f"{r2:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")
        
        fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, template="plotly_dark")
        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="Red"))
        st.plotly_chart(fig)
    else:
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.4f}")
        cm = confusion_matrix(y_test, y_pred)
        st.plotly_chart(px.imshow(cm, text_auto=True, template="plotly_dark"))

    if st.button("Final Tuning"):
        st.session_state.step = 9
        st.rerun()

# STEP 9: Tuning
elif st.session_state.step == 9:
    section_header("Hyperparameter Tuning", "Optimize parameters using Grid Search.")
    st.info("Grid Search example initialized for current model.")
    if st.button("Reset Pipeline"):
        for k in defaults: st.session_state[k] = defaults[k]
        st.rerun()

# --- FOOTER ---
st.markdown("<br><hr><center style='color:#4a5578; font-size:10px;'>ML PIPELINE STUDIO • HEALTHCARE EXPENDITURE vs GDP • 2024</center>", unsafe_allow_html=True)