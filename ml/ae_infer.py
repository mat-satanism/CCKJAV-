from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model

RAW8 = ["koi_period","koi_duration","koi_depth","koi_prad",
        "koi_steff","koi_slogg","koi_srad","koi_kepmag"]

def main():
    ap = argparse.ArgumentParser(description="AE+KMeans inference on KOI/TOI with RAW8 columns")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--artifacts", default="artifacts_ae")
    ap.add_argument("--out", default="preds/ae_clusters.csv")
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    in_path = (base / args.csv) if not Path(args.csv).is_absolute() else Path(args.csv)
    art = (base / args.artifacts) if not Path(args.artifacts).is_absolute() else Path(args.artifacts)
    out_path = (base / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scaler = load(art / "scaler.joblib")
    enc = load_model(art / "encoder_model.h5")
    km = load(art / "kmeans.joblib")

    df = pd.read_csv(in_path)
    X = df[RAW8].astype(float).replace([np.inf,-np.inf], np.nan).fillna(df[RAW8].mean())
    Xn = scaler.transform(X.values)
    Z = enc.predict(Xn, verbose=0)
    cl = km.predict(Z)
    dist = np.min(km.transform(Z), axis=1)

    out = df.copy()
    out["cluster"] = cl
    out["cluster_dist"] = dist
    out.to_csv(out_path, index=False)
    print(f"[AE] wrote → {out_path}")

if __name__ == "__main__":
    main()