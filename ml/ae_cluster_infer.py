import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

FEATURES_RAW = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
]
FEATURES_MODEL = [
    "koi_period_log", "koi_duration", "koi_depth_log", "koi_prad_log",
    "koi_steff", "koi_slogg", "koi_srad_log", "koi_kepmag",
    "koi_depth_missing", "koi_prad_missing", "koi_srad_missing",
    "koi_steff_missing", "koi_slogg_missing",
]

def _pick_feature_set(df: pd.DataFrame):
    if all(c in df.columns for c in FEATURES_MODEL):
        return FEATURES_MODEL
    if all(c in df.columns for c in FEATURES_RAW):
        return FEATURES_RAW
    raise ValueError(
        "Input must contain either model features "
        f"{FEATURES_MODEL} or RAW {FEATURES_RAW}"
    )

def main(csv: str, out: str | None = None, artifacts: str = "artifacts_ae"):
    base_dir = Path(__file__).resolve().parents[1]
    in_path = Path(csv)
    art_dir = (base_dir / artifacts) if not Path(artifacts).is_absolute() else Path(artifacts)

    # Вихід
    if out is None:
        out_path = base_dir / "preds" / "toi_ae_clusters.csv"
    else:
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = base_dir / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    (base_dir / "reports").mkdir(exist_ok=True)

    df = pd.read_csv(in_path)
    feats = _pick_feature_set(df)
    X = df[feats].copy()

    X = X.astype(float)
    X_filled = X.fillna(X.mean())
    scaler = StandardScaler().fit(X_filled.values)
    Xn = scaler.transform(X_filled.values)

    ae_path = art_dir / "autoencoder_full.h5"
    enc_path = art_dir / "encoder_model.h5"
    if ae_path.exists():
        ae = load_model(ae_path)
        try:
            if enc_path.exists():
                enc = load_model(enc_path)
                Z = enc.predict(Xn, verbose=0)
            else:
                code_layer = ae.layers[-2]
                from tensorflow.keras import Model
                enc = Model(ae.input, code_layer.output)
                Z = enc.predict(Xn, verbose=0)
        except Exception:
            Z = ae.predict(Xn, verbose=0)
    else:
        raise FileNotFoundError(f"Autoencoder didn't found: {ae_path}")

    km = KMeans(n_clusters=5, random_state=42)
    clusters = km.fit_predict(Z)
    dists = np.min(km.transform(Z), axis=1)

    df["cluster"] = clusters
    df["cluster_dist"] = dists
    df.to_csv(out_path, index=False)

    report = {
        "input_csv": str(in_path),
        "output_csv": str(out_path),
        "features_used": feats,
        "n_clusters": int(km.n_clusters),
        "cluster_centers_shape": list(np.array(km.cluster_centers_).shape),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    with open(base_dir / "reports" / "ae_infer_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Clustering completed: {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", dest="csv", type=str, required=False, help="Path to cleaned TOI CSV")
    p.add_argument("--in", dest="inp", type=str, required=False, help="Alias for --csv")
    p.add_argument("--out", dest="out", type=str, default=None)
    p.add_argument("--artifacts", dest="artifacts", type=str, default="artifacts_ae")
    args = p.parse_args()

    csv_arg = args.csv or args.inp
    if not csv_arg:
        p.error("one of --csv or --in is required")

    main(csv=csv_arg, out=args.out, artifacts=args.artifacts)
