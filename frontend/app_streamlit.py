from __future__ import annotations
import os, sys, json, io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ──────────────────────────────────────────────────────────────────────────────
# Make project root importable → we can use `ml.*`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Backend utils
from ml.mapping import RAW8, add_model_features, toi_to_koi_raw, LABEL_CANDS, normalize_labels
from ml.model_utils import load_artifacts, ensure_features

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Exoplanet Classifier", layout="wide", initial_sidebar_state="expanded")

st.title("🔭 Exoplanet Classifier")

# Sidebar
st.sidebar.header("⚙️ Settings")

default_art = ROOT / "artifacts"
art_dir_inp = st.sidebar.text_input("Artifacts path", value=str(default_art))
threshold = st.sidebar.slider("Decision threshold (for positive class)", 0.0, 1.0, 0.5, 0.01)

# ──────────────────────────────────────────────────────────────────────────────
# Cache helpers
@st.cache_resource(show_spinner=False)
def _load_artifacts_cached(path_str: str):
    return load_artifacts(Path(path_str))

@st.cache_data(show_spinner=False)
def _build_features_cached(df_in: pd.DataFrame, features: list[str]):
    return ensure_features(df_in, features)

# Try load artifacts
model = label_enc = features = metrics = None
artifacts_ok = False
try:
    model, label_enc, features, metrics = _load_artifacts_cached(art_dir_inp)
    artifacts_ok = True
except Exception as e:
    st.sidebar.error(f"Artifacts not loaded: {e}")

if artifacts_ok:
    with st.sidebar.expander("Model metrics (from metrics.json)", expanded=False):
        try:
            st.write({"roc_auc_ovr": metrics.get("roc_auc_ovr")})
            cr = metrics.get("classification_report")
            if cr:
                st.write("classification_report:")
                st.json(cr)
            st.caption(f"Label col: {metrics.get('label_col')}, classes: {metrics.get('classes')}")
        except Exception as e:
            st.warning(f"Cannot display metrics: {e}")

# ──────────────────────────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["Single Object", "Batch (CSV)"])

# =============================================================================
# TAB 1: Single Object
# =============================================================================
with tab_single:
    st.subheader("🛰️ Interactive prediction for a single candidate")
    st.caption("Enter 8 raw parameters. The model will build features, calculate probabilities, and explain details.")

    # ──────────────────────────────────────────────────────────────────────────
    # 1) 3D planet — HTML with Three.js
    # ──────────────────────────────────────────────────────────────────────────
    html_file = Path(__file__).parent / "exoplanet_3d.html"
    if not html_file.exists():
       st.error(f"File {html_file} not found.")
    else:
        with open(html_file, "r", encoding="utf-8") as f:
            exoplanet_html = f.read()

        planet_iframe = components.html(
            exoplanet_html,
            height=600,
            scrolling=True
        )


    # ──────────────────────────────────────────────────────────────────────────
    # 2) Input parameters (8 fields)
    # ──────────────────────────────────────────────────────────────────────────
    cols = st.columns(4)
    inputs = {}
    fields = [
        ("koi_period", "Orbital period (days)"),
        ("koi_duration", "Transit duration (hours)"),
        ("koi_depth", "Transit depth (ppm)"),
        ("koi_prad", "Planet radius (R_Earth)"),
        ("koi_steff", "Stellar Teff (K)"),
        ("koi_slogg", "Stellar log g"),
        ("koi_srad", "Stellar radius (R_Sun)"),
        ("koi_kepmag", "TESS/Kep magnitude"),
    ]
    for i, (key, label) in enumerate(fields):
        with cols[i % 4]:
            val = st.number_input(label, value=float("nan"), key=key)
            inputs[key] = None if np.isnan(val) else float(val)

    # ──────────────────────────────────────────────────────────────────────────
    # 3) Get values for 3D model
    # ──────────────────────────────────────────────────────────────────────────
    koi_prad_val = inputs.get("koi_prad")
    koi_steff_val = inputs.get("koi_steff")
    
    # If values are valid, display them
    if koi_prad_val is not None and koi_steff_val is not None:
        st.caption(f"🌍 Planet radius: {koi_prad_val:.2f} R⊕ | ⭐ Stellar temp: {koi_steff_val:.0f} K")

    # ──────────────────────────────────────────────────────────────────────────
    # 4) ML model prediction
    # ──────────────────────────────────────────────────────────────────────────
    if st.button("Predict", use_container_width=True, type="primary", disabled=not artifacts_ok):
        if not artifacts_ok:
            st.error("Artifacts not loaded")
        else:
            # Validate parameters for 3D
            planet_radius_safe = koi_prad_val if koi_prad_val is not None else 1.0
            stellar_teff_safe = koi_steff_val if koi_steff_val is not None else 5500
            
            # Update 3D planet model
            components.html(f"""
            <script>
            (function() {{
                // Find all iframes on the page
                const iframes = window.parent.document.querySelectorAll('iframe');
                
                // Find iframe with 3D model (first large iframe)
                iframes.forEach(iframe => {{
                    if (iframe.offsetHeight > 500) {{
                        iframe.contentWindow.postMessage({{
                            type: "update_params",
                            params: {{
                                planet_radius: {planet_radius_safe},
                                stellar_teff: {stellar_teff_safe}
                            }}
                        }}, "*");
                        console.log("Message sent to iframe:", {{planet_radius: {planet_radius_safe}, stellar_teff: {stellar_teff_safe}}});
                    }}
                }});
            }})();
            </script>
            """, height=0)
            
            # ML prediction
            raw_df = pd.DataFrame([inputs])
            model_df = add_model_features(raw_df)
            X = model_df.reindex(columns=features)
            for c in X.columns:
                if X[c].isna().any():
                    X[c] = X[c].fillna(0.0)

            proba = model.predict_proba(X.values)[0]
            classes = list(label_enc.classes_)
            pred_idx = int(np.argmax(proba))
            pred_class = label_enc.inverse_transform([pred_idx])[0]
            pred_conf = float(proba[pred_idx])

            pos_class = "PLANET" if "PLANET" in classes else classes[pred_idx]
            pos_idx   = classes.index(pos_class)
            is_pos    = proba[pos_idx] >= threshold

            k1, k2, k3 = st.columns([2,1,1])
            with k1:
                st.metric("Predicted class", pred_class)
            with k2:
                st.metric("Confidence", f"{pred_conf:.3f}")
            with k3:
                st.metric(f"Is {pos_class} @ {threshold:.2f}?", "YES" if is_pos else "NO")

            prob_df = pd.DataFrame({"class": classes, "probability": proba})
            st.write("### Class probabilities")
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

            fig = plt.figure(figsize=(5.8, 3.2))
            plt.bar(classes, proba)
            plt.ylim(0, 1)
            plt.ylabel("Probability")
            plt.title("Class probabilities")
            st.pyplot(fig, clear_figure=True)

            with st.expander("Transformed features (MODEL)", expanded=False):
                st.json({k: (None if pd.isna(v) else float(v)) for k, v in model_df.iloc[0].to_dict().items()})
            with st.expander("Raw input (RAW8)", expanded=False):
                st.json(inputs)

# =============================================================================
# TAB 2: Batch CSV
# =============================================================================
with tab_batch:
    st.subheader("📦 Batch analysis (TOI/TESS/KOI CSV)")
    up = st.file_uploader("Upload CSV", type=["csv", "txt"])
    log_box = st.checkbox("Show step-by-step logs", value=False)
    logs: list[str] = []

    def log(msg: str):
        if log_box:
            logs.append(msg)

    if up is not None and artifacts_ok:
        try:
            df_in = pd.read_csv(up, comment="#", engine="python")
            st.success(f"Loaded CSV: shape={df_in.shape}")
            st.write("Detected columns (first 40):", list(df_in.columns)[:40])

            # Build features exactly matching the trained model
            log("Building features via toi_to_koi_raw → add_model_features")
            X, model_df = _build_features_cached(df_in, features)

            log("Running inference (predict_proba)")
            P = model.predict_proba(X.values)
            classes = list(label_enc.classes_)
            pred_idx = np.argmax(P, axis=1)
            pred = label_enc.inverse_transform(pred_idx)
            maxp = P[np.arange(len(pred_idx)), pred_idx]

            out = df_in.copy()
            out["pred_class"] = pred
            out["pred_prob"] = maxp

            # per-class probability columns (optional nice-to-have)
            for i, c in enumerate(classes):
                out[f"proba_{c.replace(' ', '_').lower()}"] = P[:, i]

            # thresholded decision for pos_class
            pos_class = "PLANET" if "PLANET" in classes else classes[np.argmax(np.bincount(pred_idx))]
            pos_idx = classes.index(pos_class)
            out[f"is_{pos_class.lower()}_{threshold:.2f}"] = (P[:, pos_idx] >= threshold).astype(int)

            st.write("### Preview of predictions")
            st.dataframe(out.head(30), use_container_width=True)

            st.download_button(
                "Download predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                mime="text/csv",
                file_name="predictions.csv",
                use_container_width=True,
            )

            # ───────────────── visuals
            st.write("## Analysis")

            # 1) Histogram of model confidence
            st.write("Histogram of predicted confidences (max class)")
            fig = plt.figure(figsize=(6,4))
            plt.hist(out["pred_prob"].dropna().values, bins=30)
            plt.xlabel("max class probability")
            plt.ylabel("count")
            plt.title("Distribution of predicted confidences")
            st.pyplot(fig, clear_figure=True)

            # 2) P–R portrait (log-log scatter), TESS (pl_orbper/pl_rade) or KOI (koi_period/koi_prad)
            def pr_portrait(df: pd.DataFrame, xcol: str, ycol: str, title: str):
                x = pd.to_numeric(df[xcol], errors="coerce")
                y = pd.to_numeric(df[ycol], errors="coerce")
                m = np.isfinite(x) & np.isfinite(y)
                col = np.array(out["pred_class"] == "PLANET", dtype=int)[m] if "pred_class" in out.columns else None
                fig = plt.figure(figsize=(6,4))
                plt.scatter(x[m], y[m], c=None if col is None else col, alpha=0.6)
                plt.xscale("log"); plt.yscale("log")
                plt.xlabel(xcol); plt.ylabel(ycol); plt.title(title)
                st.pyplot(fig, clear_figure=True)

            if {"pl_orbper","pl_rade"}.issubset(out.columns):
                st.write("P–R portrait (TESS columns)")
                pr_portrait(out, "pl_orbper", "pl_rade", "P vs R_p (TESS)")
            elif {"koi_period","koi_prad"}.issubset(out.columns):
                st.write("P–R portrait (KOI columns)")
                pr_portrait(out, "koi_period", "koi_prad", "P vs R_p (KOI)")
                st.write("Correlation heatmap")
            corr_cols = [c for c in [
                "pl_orbper","pl_trandurh","pl_trandep","pl_rade","pl_insol","st_teff","st_logg","st_rad",
                "koi_period","koi_duration","koi_depth","koi_prad","koi_steff","koi_slogg","koi_srad"
            ] if c in out.columns]
            if len(corr_cols) >= 3:
                M = out[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
                fig = plt.figure(figsize=(6.5,5))
                im = plt.imshow(M.values, interpolation="nearest")
                plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
                plt.yticks(range(len(corr_cols)), corr_cols)
                plt.title("Correlation heatmap")
                plt.colorbar(im, fraction=0.046, pad=0.04)
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("Not enough numeric astro features for a heatmap (need ≥3 present columns).")

            # 4) Confusion matrix + report if ground-truth exists
            gt_col = next((c for c in LABEL_CANDS if c in out.columns), None)
            if gt_col is not None:
                st.write(f"Confusion Matrix (ground truth: {gt_col})")
                y_true = normalize_labels(out[gt_col])
                y_pred = out["pred_class"].astype(str).str.upper().str.strip()
                labels = sorted(list(set(y_true) | set(y_pred)))

                from sklearn.metrics import confusion_matrix, classification_report
                M = confusion_matrix(y_true, y_pred, labels=labels)
                fig = plt.figure(figsize=(6,5))
                plt.imshow(M, cmap="Blues")
                plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
                plt.yticks(range(len(labels)), labels)
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        plt.text(j, i, str(M[i,j]), ha="center", va="center", color="black")
                plt.title("Confusion Matrix")
                plt.colorbar(fraction=0.046, pad=0.04)
                st.pyplot(fig, clear_figure=True)

            else:
                st.info("Ground truth labels not found in uploaded CSV — confusion matrix is skipped.")
            
            # logs
            if log_box and logs:
                st.write("### Logs")
                st.code("\n".join(logs))

        except Exception as e:
            st.error(f"Failed to process CSV: {e}")
    elif up is None:
        st.info("Upload a CSV to run batch analysis.")
    elif not artifacts_ok:
        st.error("Artifacts not loaded — check path in the sidebar.")

# ──────────────────────────────────────────────────────────────────────────────
st.caption(f"Artifacts expected at:  **{art_dir_inp}**")
