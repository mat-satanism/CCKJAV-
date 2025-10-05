import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from .model_utils import load_artifacts, ensure_features
from .mapping import LABEL_CANDS, normalize_labels

def main():
    ap = argparse.ArgumentParser(description="Infer on TOI/TESS CSV (raw NASA export OK)")
    ap.add_argument("--csv", required=True, help="data/raw/TESS.csv")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default=None, help="preds/predictions_tess.csv")
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    in_path  = (Path(args.csv) if Path(args.csv).is_absolute() else base/args.csv)
    out_path = Path(args.out) if args.out else (base / "preds" / "predictions_tess.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, le, features, _ = load_artifacts(base / args.artifacts)

    df_in = pd.read_csv(in_path, comment="#", engine="python")
    X, feat_df = ensure_features(df_in, features)

    P = model.predict_proba(X.values)
    pred_idx = np.argmax(P, axis=1)
    if len(le.classes_) == 2:
        classes = list(le.classes_)
    else:
        classes = list(le.classes_)
    pred_cls = le.inverse_transform(pred_idx)
    maxp = P[np.arange(len(pred_idx)), pred_idx]

    out = df_in.copy()
    out["pred_class"] = pred_cls
    out["pred_prob"]  = maxp
    if "PLANET" in classes:
        pidx = classes.index("PLANET")
        out["proba_planet"] = P[:, pidx]

    gt_col = next((c for c in LABEL_CANDS if c in df_in.columns), None)
    if gt_col:
        out["label_norm"] = normalize_labels(df_in[gt_col])

    out.to_csv(out_path, index=False)
    print(f"[OK] Wrote predictions → {out_path}")

if __name__ == "__main__":
    main()