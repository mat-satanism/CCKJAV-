from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model
from tensorflow import keras

RAW_FEATURES = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
]
MODEL_FEATURES = [
    "koi_period_log", "koi_duration", "koi_depth_log", "koi_prad_log",
    "koi_steff", "koi_slogg", "koi_srad_log", "koi_kepmag",
    "koi_depth_missing", "koi_prad_missing", "koi_srad_missing",
    "koi_steff_missing", "koi_slogg_missing",
]


def has_all(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


def reconstruct_raw_from_model(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # Safe helpers
    def get(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    out["koi_period"]   = np.expm1(get("koi_period_log"))
    out["koi_duration"] = get("koi_duration")
    out["koi_depth"]    = np.expm1(get("koi_depth_log"))
    out["koi_prad"]     = np.expm1(get("koi_prad_log"))
    out["koi_steff"]    = get("koi_steff")
    out["koi_slogg"]    = get("koi_slogg")
    out["koi_srad"]     = np.expm1(get("koi_srad_log"))
    out["koi_kepmag"]   = get("koi_kepmag")
    return out


def main():
    ap = argparse.ArgumentParser(description="AE+KMeans inference (consistent with ae_train.py)")
    ap.add_argument("--csv", required=True, help="Path to input CSV (RAW8 or MODEL features)")
    ap.add_argument("--out", default=None, help="Output CSV path (default: preds/<stem>_ae_clusters.csv)")
    ap.add_argument("--artifacts", default="artifacts_ae", help="AE artifacts folder")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    in_path = Path(args.csv)
    out_path = Path(args.out) if args.out else (base_dir / "preds" / (in_path.stem + "_ae_clusters.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    art = Path(args.artifacts)
    scaler = load(art / "scaler.joblib")
    kmeans = load(art / "kmeans.joblib")

    enc_path = art / "encoder_model.h5"
    if enc_path.exists():
        encoder = load_model(enc_path)
    else:
        ae = load_model(art / "autoencoder_full.h5")
        encoder = keras.Model(ae.input, ae.layers[-2].output)

    df = pd.read_csv(in_path)

    if has_all(df, RAW_FEATURES):
        raw = df[RAW_FEATURES].copy()
        used_feats = RAW_FEATURES
    elif has_all(df, MODEL_FEATURES):
        raw = reconstruct_raw_from_model(df)
        used_feats = [f"reconstructed:{c}" for c in RAW_FEATURES]
    else:
        raise ValueError("Input must contain RAW8 or MODEL features; neither set is complete.")

    X = raw.astype(float).replace([np.inf, -np.inf], np.nan)
    X_filled = X.fillna(X.mean())

    Xn = scaler.transform(X_filled.values)
    Z = encoder.predict(Xn, verbose=0)

    clusters = kmeans.predict(Z)
    dists = np.min(kmeans.transform(Z), axis=1)

    out = df.copy()
    out["cluster"] = clusters
    out["cluster_dist"] = dists
    out.to_csv(out_path, index=False)

    cfg_path = art / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    report = {
        "input_csv": str(in_path),
        "output_csv": str(out_path),
        "features_used": used_feats,
        "n_clusters": int(cfg.get("n_clusters", getattr(kmeans, "n_clusters", -1))),
    }
    (base_dir / "reports").mkdir(exist_ok=True)
    (base_dir / "reports" / "ae_infer_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[AE] Clustering completed → {out_path}")


if __name__ == "__main__":
    main()