from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load

# allow "python ml/infer_tess.py" as well:
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.mapping import toi_to_koi_raw
from ml.features import add_model_features  # only to derive raw->model columns

def load_features_from_artifacts(art: Path) -> list[str]:
    fpath = art / "features.json"
    if not fpath.exists():
        raise FileNotFoundError(f"features.json not found in {art}")
    return json.loads(fpath.read_text(encoding="utf-8"))

def load_members_from_blend(art: Path) -> tuple[list[str], float] | tuple[None, None]:
    bpath = art / "blend.json"
    if not bpath.exists():
        return None, None
    cfg = json.loads(bpath.read_text(encoding="utf-8"))
    members = cfg.get("members", [])
    thr = float(cfg.get("threshold", 0.5))
    return members, thr

def main():
    ap = argparse.ArgumentParser(description="Inference on TESS/TOI CSV (strict features from training)")
    ap.add_argument("--csv", required=True, help="Path to TESS.csv / tess_cleaned.csv")
    ap.add_argument("--artifacts", default="artifacts", help="Folder with saved models & features.json")
    ap.add_argument("--out", default=None, help="Output CSV path")
    ap.add_argument("--thr", type=float, default=None, help="Override threshold (default from blend.json or 0.5)")
    args = ap.parse_args()

    in_path = Path(args.csv)
    art = Path(args.artifacts)
    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_preds.csv")

    # 1) Read input (archive TESS has '#' comments)
    df_in = pd.read_csv(in_path, comment="#", engine="python")

    # 2) Map to KOI-like raw and derive model features
    raw = toi_to_koi_raw(df_in)
    feat_full = add_model_features(raw)  # may include optional extras

    # 3) Load EXACT feature list used in training
    FEATURES = load_features_from_artifacts(art)

    # 4) Build X with EXACT columns & order; fill missing with 0.0
    #    (ignore any extra columns present in feat_full)
    missing = [c for c in FEATURES if c not in feat_full.columns]
    if missing:
        # Try to create missing columns as 0.0 (some features might be optional flags from training)
        for c in missing:
            feat_full[c] = 0.0
        still_missing = [c for c in FEATURES if c not in feat_full.columns]
        if still_missing:
            raise KeyError(f"Cannot construct required features for inference: {still_missing}")

    X = feat_full[FEATURES].astype(float).fillna(0.0)

    # 5) Try ensemble first; else fallback to single model.joblib
    members, thr_from_blend = load_members_from_blend(art)
    if args.thr is not None:
        threshold = float(args.thr)
    elif thr_from_blend is not None:
        threshold = thr_from_blend
    else:
        threshold = 0.5

    probs = None
    used_members = []

    if members:
        for name in members:
            model_path = art / f"model_{name}.joblib"
            if not model_path.exists():
                print(f"[WARN] Missing {model_path}, skipping this member")
                continue
            mdl = load(model_path)
            p = mdl.predict_proba(X)[:, 1]
            probs = p if probs is None else (probs + p)
            used_members.append(name)
        if probs is None:
            print("[WARN] No ensemble members found; falling back to model.joblib")
    if probs is None:
        single_model = art / "model.joblib"
        if not single_model.exists():
            raise FileNotFoundError(f"No model found: {single_model} or ensemble members in {art}")
        mdl = load(single_model)
        probs = mdl.predict_proba(X)[:, 1]
        used_members = ["single:model.joblib"]
    else:
        probs = probs / len(used_members)

    pred = np.where(probs >= threshold, "PLANET", "FALSE POSITIVE")

    # 6) Save output (attach a few informative columns when present)
    out = df_in.copy()
    for c in ["toi","tid","tfopwg_disp","pl_orbper","pl_trandurh","pl_trandep","pl_rade","pl_insol","st_tmag","st_teff","st_logg","st_rad"]:
        if c in df_in.columns and c not in out.columns:
            out[c] = df_in[c]
    out["proba_planet"] = probs
    out["pred_class"] = pred

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")

    n_planets = int((pred == "PLANET").sum())
    print(f"[OK] Wrote → {out_path}")
    print(f"   models: {used_members}")
    print(f"   threshold: {threshold:.6f}")
    print(f"   rows: {len(out)}, predicted PLANET: {n_planets} ({n_planets/len(out):.1%})")

if __name__ == "__main__":
    main()