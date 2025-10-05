from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .model_utils import load_artifacts, ensure_features
from .mapping import LABEL_CANDS, normalize_labels

def main():
    ap = argparse.ArgumentParser(description="Batch predict on CSV (TOI/TESS or KOI-like)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default=None)
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    in_path  = (Path(args.csv) if Path(args.csv).is_absolute() else base/args.csv)
    out_path = Path(args.out) if args.out else (base/"preds"/(Path(args.csv).stem+"_preds.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.log: print(f"[i] reading {in_path}")

    model, le, features, _ = load_artifacts(base / args.artifacts)
    df_in = pd.read_csv(in_path, comment="#", engine="python")
    if args.log: print(f"[i] columns: {list(df_in.columns)[:30]}")

    X, _ = ensure_features(df_in, features)
    P = model.predict_proba(X.values)
    pred_idx = np.argmax(P, axis=1)
    pred_cls = le.inverse_transform(pred_idx)
    maxp = P[np.arange(len(pred_idx)), pred_idx]

    out = df_in.copy()
    out["pred_class"] = pred_cls
    out["pred_prob"]  = maxp

    gt_col = next((c for c in LABEL_CANDS if c in df_in.columns), None)
    if gt_col:
        out["label_norm"] = normalize_labels(df_in[gt_col])

    out.to_csv(out_path, index=False)
    if args.log: print(f"[OK] wrote → {out_path}")

if __name__ == "__main__":
    main()