# ml/app_streamlit.py
from __future__ import annotations
import json, io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from joblib import load

from mapping import RAW8, toi_to_koi_raw, add_model_features, LABEL_CANDS, normalize_labels

st.set_page_config(page_title="Exoplanet Classifier (KOI→TESS)", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def log(msg: str):
    st.session_state.setdefault("log", [])
    st.session_state["log"].append(msg)

def load_artifacts(art_dir: Path):
    model = load(art_dir / "model.joblib")
    label_enc = load(art_dir / "label_encoder.joblib")
    features = json.loads((art_dir / "features.json").read_text())
    metrics = json.loads((art_dir / "metrics.json").read_text())
    return model, label_enc, features, metrics

def prepare_X_from_raw(raw_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = add_model_features(raw_df)
    X = df.reindex(columns=features)
    # заповнюємо NaN нулями (бо є флаги *_missing)
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(0.0)
    return X, df

def figure_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def bar_probs(classes: list[str], probs: np.ndarray, title: str):
    fig = plt.figure(figsize=(4.5,3.2))
    plt.bar(classes, probs)
    plt.ylim(0,1)
    plt.ylabel("Probability")
    plt.title(title)
    return fig

def hist_max_proba(maxp: np.ndarray, title="Histogram of Max Probability"):
    fig = plt.figure(figsize=(6,3.2))
    plt.hist(maxp, bins=25)
    plt.xlabel("Max class probability")
    plt.ylabel("Count")
    plt.title(title)
    return fig

def confusion_if_labels(y_true: pd.Series, y_pred: np.ndarray, ordered):
    from sklearn.metrics import confusion_matrix, classification_report
    M = confusion_matrix(y_true, y_pred, labels=ordered)
    fig = plt.figure(figsize=(5,4))
    plt.imshow(M, cmap="Blues")
    plt.xticks(range(len(ordered)), ordered, rotation=45, ha="right")
    plt.yticks(range(len(ordered)), ordered)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            plt.text(j, i, str(M[i,j]), ha="center", va="center", color="black")
    plt.title("Confusion Matrix on provided labels")
    plt.colorbar(fraction=0.046, pad=0.04)
    return fig, classification_report(y_true, y_pred, labels=ordered, output_dict=True)

# ---------------------------
# Sidebar (artifacts & options)
# ---------------------------
base_dir = Path(__file__).resolve().parents[1]
st.sidebar.subheader("Artifacts")
art_dir = Path(st.sidebar.text_input("Path to artifacts", str(base_dir / "artifacts")))
threshold = st.sidebar.slider("High-confidence threshold", 0.5, 0.99, 0.80, 0.01)

# Load artifacts
try:
    model, label_enc, FEATURES, METR = load_artifacts(art_dir)
    st.sidebar.success("Artifacts loaded")
except Exception as e:
    st.sidebar.error(f"Failed to load artifacts: {e}")
    st.stop()

CLASSES = list(label_enc.classes_)

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2 = st.tabs(["Single Object (8 features)", "Batch CSV (TOI/TESS)"])

# ===========================
# Tab 1 — Single
# ===========================
with tab1:
    st.header("Interactive prediction for a single object")
    cols = st.columns(4)
    vals = {}
    nice = {
        "koi_period":   "Orbital period (days)",
        "koi_duration": "Transit duration (hours)",
        "koi_depth":    "Transit depth (ppm)",
        "koi_prad":     "Planet radius (R_Earth)",
        "koi_steff":    "Star Teff (K)",
        "koi_slogg":    "Star logg",
        "koi_srad":     "Star radius (R_Sun)",
        "koi_kepmag":   "TESS/Kep magnitude",
    }
    for i, k in enumerate(RAW8):
        with cols[i % 4]:
            vals[k] = st.number_input(nice[k], value=float("nan"))

    if st.button("Predict", type="primary"):
        log("[single] Building raw dataframe…")
        raw_df = pd.DataFrame([{k: (None if np.isnan(v) else v) for k, v in vals.items()}])

        # Build model features
        X, full_df = prepare_X_from_raw(raw_df, FEATURES)
        log("[single] Features prepared")

        # Predict
        proba = model.predict_proba(X.values)[0]
        pred_idx = int(np.argmax(proba))
        pred_cls = label_enc.inverse_transform([pred_idx])[0]
        st.subheader(f"Predicted class: **{pred_cls}**")

        # Probabilities table + bar
        st.write(pd.DataFrame({"class": CLASSES, "probability": proba}))
        st.image(figure_to_bytes(bar_probs(CLASSES, proba, "Class probabilities")))

        # Details: transformed features
        with st.expander("Show transformed features (what went into the model)"):
            st.dataframe(full_df.reindex(columns=FEATURES))

        # Model metrics (from training)
        st.markdown("### Model metrics (from training)")
        colm = st.columns(3)
        with colm[0]:
            st.metric("ROC-AUC (OVR)", f"{METR.get('roc_auc_ovr', None)}")
        with colm[1]:
            st.write("Classes:", METR.get("classes", []))
        with colm[2]:
            st.write("Label column:", METR.get("label_col", "—"))
        st.write("Classification report (train holdout):")
        st.json(METR.get("classification_report", {}))

        # High-confidence flag
        st.info(f"High-confidence (>{threshold:.2f})? → **{proba[pred_idx] >= threshold}** (p={proba[pred_idx]:.3f})", icon="✅")

        # Log
        with st.expander("Logs"):
            st.code("\n".join(st.session_state.get("log", [])) or "<no logs>")

# ===========================
# Tab 2 — Batch
# ===========================
with tab2:
    st.header("Batch predictions for TOI/TESS CSV")
    f = st.file_uploader("Upload TOI/TESS CSV (NASA archive export is OK)", type=["csv","txt"])
    if f is not None:
        try:
            log("[batch] Reading CSV (skipping commented lines)…")
            df_in = pd.read_csv(f, comment="#", engine="python")
            st.write("Detected columns:", list(df_in.columns)[:40])

            log("[batch] Mapping to RAW8…")
            raw_df = toi_to_koi_raw(df_in)

            log("[batch] Building model features…")
            X, feat_df = prepare_X_from_raw(raw_df, FEATURES)

            log("[batch] Running inference…")
            P = model.predict_proba(X.values)
            pred_idx = np.argmax(P, axis=1)
            pred_cls = label_enc.inverse_transform(pred_idx)
            maxp = P[np.arange(len(pred_idx)), pred_idx]

            out = df_in.copy()
            out["pred_class"] = pred_cls
            out["pred_prob"]  = maxp

            # показати кілька рядків
            st.success(f"Predictions computed for {len(out)} rows")
            st.dataframe(out.head(30))

            # скачування
            st.download_button(
                "Download predictions CSV",
                out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

            # ----- графіки аналізу -----
            st.subheader("Analysis visuals")

            c1, c2 = st.columns(2)
            with c1:
                st.image(figure_to_bytes(hist_max_proba(maxp)))
            with c2:
                # частка високої впевненості по передбачених класах
                prop = (
                    pd.Series(pred_cls)
                      .groupby(pred_cls)
                      .apply(lambda idx: float(np.mean(maxp[pred_cls==idx.name] >= threshold)))
                      .reindex(CLASSES)
                      .fillna(0.0)
                )
                fig = plt.figure(figsize=(6,3.2))
                plt.bar(prop.index, prop.values)
                plt.ylim(0,1)
                plt.title(f"Proportion above {threshold:.2f} by predicted class")
                st.image(figure_to_bytes(fig))

            # якщо у файлі є істинні мітки — покажемо матрицю
            gt_col = next((c for c in LABEL_CANDS if c in df_in.columns), None)
            if gt_col:
                st.subheader("Provided labels found — evaluation on this CSV")
                y_true = normalize_labels(df_in[gt_col])
                fig, report = confusion_if_labels(y_true, pred_cls, ordered=CLASSES)
                st.image(figure_to_bytes(fig))
                st.json(report)

            # логування
            with st.expander("Logs"):
                st.code("\n".join(st.session_state.get("log", [])) or "<no logs>")

        except Exception as e:
            st.error(f"Failed: {e}")
            with st.expander("Logs"):
                st.code("\n".join(st.session_state.get("log", [])) or "<no logs>")
