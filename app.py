"""Streamlit app for the pre-trained mushroom classifiers.

Two ways to use it:
    1. Predict a single mushroom by choosing its traits from dropdowns.
    2. Upload a labeled test CSV and evaluate / compare the models.

Models are loaded from ./model and are never retrained here.
"""

import html
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_TEST_FILE = BASE_DIR / "test_data.csv"
MAX_UPLOAD_ROWS = 50_000

st.set_page_config(page_title="Mushroom Classifier Lab", layout="wide")

# ------------------------------------------------------------ trait labels
# Human-readable names for the single-letter codes (UCI Mushroom documentation).
COLOR_NAMES = {
    "n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "r": "green", "o": "orange",
    "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow", "k": "black",
    "h": "chocolate",
}
SURFACE_NAMES = {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"}
CODE_NAMES = {
    "cap-shape": {"b": "bell", "c": "conical", "x": "convex", "f": "flat", "k": "knobbed",
                  "s": "sunken"},
    "cap-surface": {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"},
    "cap-color": COLOR_NAMES,
    "bruises": {"t": "yes", "f": "no"},
    "odor": {"a": "almond", "l": "anise", "c": "creosote", "y": "fishy", "f": "foul",
             "m": "musty", "n": "none", "p": "pungent", "s": "spicy"},
    "gill-attachment": {"a": "attached", "d": "descending", "f": "free", "n": "notched"},
    "gill-spacing": {"c": "close", "w": "crowded", "d": "distant"},
    "gill-size": {"b": "broad", "n": "narrow"},
    "gill-color": COLOR_NAMES,
    "stalk-shape": {"e": "enlarging", "t": "tapering"},
    "stalk-root": {"b": "bulbous", "c": "club", "u": "cup", "e": "equal", "z": "rhizomorphs",
                   "r": "rooted", "?": "unknown / missing"},
    "stalk-surface-above-ring": SURFACE_NAMES,
    "stalk-surface-below-ring": SURFACE_NAMES,
    "stalk-color-above-ring": COLOR_NAMES,
    "stalk-color-below-ring": COLOR_NAMES,
    "veil-color": COLOR_NAMES,
    "ring-number": {"n": "none", "o": "one", "t": "two"},
    "ring-type": {"c": "cobwebby", "e": "evanescent", "f": "flaring", "l": "large",
                  "n": "none", "p": "pendant", "s": "sheathing", "z": "zone"},
    "spore-print-color": COLOR_NAMES,
    "population": {"a": "abundant", "c": "clustered", "n": "numerous", "s": "scattered",
                   "v": "several", "y": "solitary"},
    "habitat": {"g": "grasses", "l": "leaves", "m": "meadows", "p": "paths", "u": "urban",
                "w": "waste", "d": "woods"},
}
FEATURE_GROUPS = {
    "Cap": ["cap-shape", "cap-surface", "cap-color"],
    "Gills": ["gill-attachment", "gill-spacing", "gill-size", "gill-color"],
    "Stalk": ["stalk-shape", "stalk-root", "stalk-surface-above-ring",
              "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring"],
    "Veil and ring": ["veil-color", "ring-number", "ring-type"],
    "Smell, spores and location": ["bruises", "odor", "spore-print-color", "population",
                                   "habitat"],
}

# ----------------------------------------------------------------- styling
# Forest and earth palette: greens, moss, tan and brown.
METRIC_COLORS = {
    "Accuracy": "#DCE8D0",
    "AUC": "#EADFC8",
    "Precision": "#D3E4CD",
    "Recall": "#E6D8C3",
    "F1": "#CFDCC2",
    "MCC": "#E3D3BC",
}
MODEL_COLORS = {
    "Logistic Regression": "#7FA36B",
    "Decision Tree": "#A67C52",
    "KNN": "#5E8C61",
    "Naive Bayes": "#C2A878",
    "Support Vector Machine": "#4F7942",
}
TABLE_CMAP = LinearSegmentedColormap.from_list("earth_rg", ["#E8C9A8", "#F5F0DC", "#B9D7A8"])
# Brighter palette for the comparison bar chart so the bars stand out.
BAR_COLORS = {
    "Accuracy": "#2D5A27",
    "AUC": "#4F7942",
    "Precision": "#8DB36B",
    "Recall": "#C2A878",
    "F1": "#A67C52",
    "MCC": "#6B4226",
}


def tint_cmap(color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("earth", ["#FFFFFF", color])


st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F3EFE2 0%, #E9EFDF 50%, #E2EAD6 100%);
    }
    .hero {
        background: linear-gradient(120deg, #1F3D2B 0%, #3E7B4F 45%, #8A6A3D 100%);
        color: white; padding: 1.4rem 1.8rem; border-radius: 16px;
        margin-bottom: 1rem; box-shadow: 0 6px 18px rgba(31, 61, 43, 0.3);
    }
    .hero h1 { color: white; margin: 0 0 0.4rem 0; font-size: 2.1rem; }
    .hero p { margin: 0; font-size: 1.02rem; opacity: 0.95; }
    .metric-card {
        border-radius: 14px; padding: 0.9rem 1rem; color: #2F3A2F; text-align: center;
        border: 1px solid rgba(0, 0, 0, 0.05); box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .metric-card .label { font-size: 0.8rem; letter-spacing: 0.06em; opacity: 0.75; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; }
    .source-pill {
        display: inline-block; background: #E4EBD9; color: #2F3A2F;
        padding: 0.25rem 0.8rem; border-radius: 999px; font-size: 0.85rem;
    }
    .result-card {
        border-radius: 16px; padding: 1.2rem 1.5rem; color: #2F3A2F;
        border-left: 8px solid; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
    }
    .result-card h2 { margin: 0 0 0.3rem 0; color: #2F3A2F; }
    .result-card p { margin: 0; }
    .edible { background: #DCEBD3; border-color: #5E8C61; }
    .poisonous { background: #F2DCD2; border-color: #A0522D; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #3E7B4F; }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: float) -> str:
    """HTML for one metric tile. Inputs are internal (labels, floats)."""
    shown = "n/a" if pd.isna(value) else f"{value:.4f}"
    color = METRIC_COLORS.get(label, "#EEEEEE")
    return (
        f'<div class="metric-card" style="background: {color};">'
        f'<div class="label">{label.upper()}</div><div class="value">{shown}</div></div>'
    )


# ------------------------------------------------------------------ helpers
@st.cache_resource
def load_artifacts():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in meta["models"].items()}
    return meta, models


@st.cache_data
def load_training_metrics():
    return pd.read_csv(MODEL_DIR / "metrics.csv", index_col=0)


@st.cache_data
def default_choices(features: tuple) -> dict:
    """Most common value of each trait in the bundled test data (form defaults)."""
    if not SAMPLE_TEST_FILE.exists():
        return {}
    sample = pd.read_csv(SAMPLE_TEST_FILE, dtype=str)
    return {f: sample[f].mode().iloc[0] for f in features if f in sample}


def trait_options(model, features: list) -> dict:
    """Allowed values per trait = categories the one-hot encoder saw in training."""
    encoder = model.named_steps["encode"]
    return {f: list(cats) for f, cats in zip(features, encoder.categories_)}


def code_label(feature: str, code: str) -> str:
    name = CODE_NAMES.get(feature, {}).get(code, code)
    return f"{name} ({code})"


def encode_target(series: pd.Series, positive_label: str) -> pd.Series | None:
    """Map target to 1 (poisonous) / 0 (edible). Accepts 'p'/'e' or 1/0."""
    values = series.astype(str).str.strip().str.lower()
    if set(values.unique()) <= {"p", "e"}:
        return (values == positive_label).astype(int)
    if set(values.unique()) <= {"0", "1"}:
        return values.astype(int)
    return None


def positive_score(model, X: pd.DataFrame) -> np.ndarray:
    # SVM is trained without probability=True, so fall back to decision_function.
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X: pd.DataFrame, y: pd.Series):
    pred = model.predict(X)
    scores = positive_score(model, X)
    auc = roc_auc_score(y, scores) if y.nunique() == 2 else np.nan
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return metrics, pred


def confidence_text(model, row: pd.DataFrame, pred: int) -> str:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0, pred]
        return f"{proba:.1%} probability"
    margin = abs(float(model.decision_function(row)[0]))
    return f"margin {margin:.2f} from the SVM boundary"


def plot_confusion(y, pred, title, color):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    fig.patch.set_alpha(0)
    ax.imshow(cm, cmap=tint_cmap(color))
    labels = ["edible", "poisonous"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            dark_cell = cm[i, j] > cm.max() / 2
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=13,
                    color="#FFFFFF" if dark_cell else "#2F3A2F")
    fig.tight_layout()
    return fig


def load_test_data(uploaded, use_sample: bool, features: list, target: str, positive: str):
    """Return (data, X, y, source, message). data is None when nothing usable is loaded."""
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded, dtype=str)
        except Exception:
            return None, None, None, None, "Could not parse the uploaded file as CSV."
        source = f"Uploaded file: {uploaded.name}"
    elif use_sample and SAMPLE_TEST_FILE.exists():
        data = pd.read_csv(SAMPLE_TEST_FILE, dtype=str)
        source = "Bundled test_data.csv"
    else:
        return None, None, None, None, "Upload a test CSV from the sidebar to evaluate models."

    data.columns = [c.strip() for c in data.columns]
    # Public copies of this dataset (Kaggle/UCI) name the target "class".
    if target not in data.columns and "class" in data.columns:
        data = data.rename(columns={"class": target})
    missing = [c for c in features + [target] if c not in data.columns]
    if missing:
        return None, None, None, None, f"The CSV is missing required columns: {', '.join(missing)}"
    if data.empty:
        return None, None, None, None, "The CSV has no rows."
    if len(data) > MAX_UPLOAD_ROWS:
        return None, None, None, None, f"Too many rows ({len(data)}). Limit is {MAX_UPLOAD_ROWS}."

    data = data.dropna(subset=[target])
    y = encode_target(data[target], positive)
    if y is None:
        return None, None, None, None, f"Target column `{target}` must contain only 'p'/'e' or 1/0."
    X = data[features].fillna("?").apply(lambda s: s.str.strip().str.lower())
    return data, X, y, source, None


# -------------------------------------------------------------- load models
try:
    meta, models = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found in ./model. Run `python train_models.py` first.")
    st.stop()

FEATURES = meta["features"]
TARGET = meta["target"]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1. Model")
    model_name = st.selectbox("Choose a classifier", list(models.keys()))

    st.header("2. Test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    if SAMPLE_TEST_FILE.exists():
        st.download_button(
            "Download sample test_data.csv",
            data=SAMPLE_TEST_FILE.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )
    use_sample = st.checkbox("Use bundled test_data.csv if nothing is uploaded", value=True)

    st.caption(f"Models trained with scikit-learn {meta.get('sklearn_version', '?')}")

model = models[model_name]

# --------------------------------------------------------------------- body
st.markdown(
    """
    <div class="hero">
      <h1>Mushroom Classifier Lab</h1>
      <p>Predict whether a mushroom is <b>edible</b> or <b>poisonous</b> from 21 categorical
      traits (UCI Mushroom dataset). Describe a single mushroom, or upload a labeled test CSV
      to evaluate and compare five pre-trained models. Models are never retrained here.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_single, tab_compare, tab_data = st.tabs(
    ["Predict a mushroom", "Evaluate on test data", "Compare all models", "Data preview"]
)

# ------------------------------------------------------ predict one mushroom
with tab_predict:
    st.subheader("Describe the mushroom")
    st.caption(f"Pick each trait below, then press Predict. Model in use: **{model_name}** "
               "(change it in the sidebar).")

    options = trait_options(model, FEATURES)
    defaults = default_choices(tuple(FEATURES))
    grouped = [f for group in FEATURE_GROUPS.values() for f in group]
    groups = dict(FEATURE_GROUPS)
    leftovers = [f for f in FEATURES if f not in grouped]
    if leftovers:
        groups["Other"] = leftovers

    choices = {}
    with st.form("predict_form"):
        for group, feats in groups.items():
            feats = [f for f in feats if f in options]
            if not feats:
                continue
            st.markdown(f"**{group}**")
            cols = st.columns(3)
            for i, feat in enumerate(feats):
                values = options[feat]
                default = defaults.get(feat)
                index = values.index(default) if default in values else 0
                choices[feat] = cols[i % 3].selectbox(
                    feat.replace("-", " ").capitalize(), values, index=index,
                    format_func=lambda code, f=feat: code_label(f, code), key=f"trait_{feat}",
                )
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    if submitted:
        row = pd.DataFrame([choices], columns=FEATURES)
        pred = int(model.predict(row)[0])
        label = "Poisonous" if pred == 1 else "Edible"
        css = "poisonous" if pred == 1 else "edible"
        st.markdown(
            f'<div class="result-card {css}"><h2>{label}</h2>'
            f"<p>{html.escape(model_name)} predicts this mushroom is <b>{label.lower()}</b> "
            f"({html.escape(confidence_text(model, row, pred))}).</p></div>",
            unsafe_allow_html=True,
        )
        st.write("")

        st.markdown("**What do all five models say?**")
        votes = []
        for name, m in models.items():
            p = int(m.predict(row)[0])
            votes.append({"Model": name, "Prediction": "poisonous" if p else "edible",
                          "Confidence": confidence_text(m, row, p)})
        votes_df = pd.DataFrame(votes).set_index("Model")
        st.dataframe(votes_df.style.map(
            lambda v: "background-color: #F2DCD2" if v == "poisonous"
            else "background-color: #DCEBD3" if v == "edible" else "",
            subset=["Prediction"],
        ))

        st.warning(
            "This is a classroom model trained on one historical dataset. "
            "Never use it to decide whether a real wild mushroom is safe to eat."
        )

# ---------------------------------------------------- evaluate on test data
data, X, y_true, source, message = load_test_data(
    uploaded, use_sample, FEATURES, TARGET, meta["positive_label"]
)
pred = None

with tab_single:
    if data is None:
        st.info(message)
    else:
        st.markdown(
            f'<span class="source-pill">{html.escape(source)} · {len(data)} rows · '
            f"{len(FEATURES)} features</span>",
            unsafe_allow_html=True,
        )
        metrics, pred = evaluate(model, X, y_true)
        st.subheader(model_name)

        cols = st.columns(6)
        for col, (label, value) in zip(cols, metrics.items()):
            col.markdown(metric_card(label, value), unsafe_allow_html=True)
        st.write("")

        color = MODEL_COLORS.get(model_name, "#D3E4CD")
        left, right = st.columns([1, 1.3])
        with left:
            st.markdown("**Confusion matrix**")
            st.pyplot(plot_confusion(y_true, pred, model_name, color))
        with right:
            st.markdown("**Classification report**")
            report = classification_report(
                y_true, pred, labels=[0, 1], target_names=["edible", "poisonous"],
                output_dict=True, zero_division=0,
            )
            st.dataframe(pd.DataFrame(report).T.round(4).style.background_gradient(
                cmap=tint_cmap(color), subset=["precision", "recall", "f1-score"],
                vmin=0.9, vmax=1.0,
            ))

        if pd.isna(metrics["AUC"]):
            st.warning("AUC needs both classes in the test data, so it is not shown.")

# --------------------------------------------------------------- comparison
with tab_compare:
    if data is None:
        st.info(message)
    else:
        st.subheader("All models on this test data")
        rows = {name: evaluate(m, X, y_true)[0] for name, m in models.items()}
        comparison = pd.DataFrame(rows).T.round(4)
        comparison.index.name = "ML Model Name"
        st.dataframe(comparison.style.background_gradient(cmap=TABLE_CMAP, axis=0)
                     .highlight_max(axis=0, props="font-weight: bold;"))

        fig, ax = plt.subplots(figsize=(10, 3.8))
        fig.patch.set_alpha(0)
        comparison.plot.bar(ax=ax, rot=0, edgecolor="white", linewidth=0.8, width=0.8,
                            color=[BAR_COLORS[c] for c in comparison.columns])
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", color="#E6E0D0", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(max(0.0, np.nanmin(comparison.values) - 0.05), 1.005)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize=9,
                  frameon=False)
        ax.set_title("Metric comparison")
        fig.tight_layout()
        st.pyplot(fig)

        with st.expander("Reference: metrics, tuned params, and 5-fold CV F1 recorded at "
                         "training time"):
            st.dataframe(load_training_metrics())

# ------------------------------------------------------------------ preview
with tab_data:
    if data is None:
        st.info(message)
    else:
        preview = data.copy()
        preview["prediction"] = np.where(pred == 1, "p", "e")
        st.dataframe(preview.head(200))
        st.download_button(
            f"Download predictions ({model_name})",
            data=preview.to_csv(index=False).encode(),
            file_name="predictions.csv",
            mime="text/csv",
        )
