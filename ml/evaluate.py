from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend для збереження PNG без GUI
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix

def main():
    ap = argparse.ArgumentParser(description="Quick evaluation visuals on KOI holdout-like splits (saved probs) or TESS preds")
    ap.add_argument("--preds", required=True, help="CSV з колонками proba_planet, pred_class (наприклад, TESS_preds)")
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # 1) Гістограма
    plt.figure(figsize=(6,4))
    plt.hist(df["proba_planet"].dropna().values, bins=30)
    plt.xlabel("proba_planet")
    plt.ylabel("count")
    plt.title("Distribution of PLANET probabilities")
    plt.tight_layout()
    plt.savefig(out / "hist_proba.png", dpi=180); plt.close()

    # 2) P–R портрет
    if {"pl_orbper","pl_rade"}.issubset(df.columns):
        x = pd.to_numeric(df["pl_orbper"], errors="coerce")
        y = pd.to_numeric(df["pl_rade"], errors="coerce")
        m = np.isfinite(x) & np.isfinite(y)
        c = (df["pred_class"] == "PLANET").astype(int).values
        plt.figure(figsize=(6,4))
        plt.scatter(x[m], y[m], c=c[m], alpha=0.6)
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Orbital period (days)")
        plt.ylabel("Planet radius (R_Earth)")
        plt.title("P vs R_p (colored by prediction)")
        plt.tight_layout()
        plt.savefig(out / "portrait_P_vs_R.png", dpi=180); plt.close()

    # 3) Heatmap кореляцій
    corr_cols = [c for c in ["pl_orbper","pl_trandurh","pl_trandep","pl_rade","pl_insol","st_teff","st_logg","st_rad"] if c in df.columns]
    if len(corr_cols) >= 3:
        M = df[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
        fig = plt.figure(figsize=(5,4))
        im = plt.imshow(M.values, interpolation="nearest")
        plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
        plt.yticks(range(len(corr_cols)), corr_cols)
        plt.title("Correlation heatmap")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out / "corr_heatmap.png", dpi=180); plt.close()

    print("[OK] Saved visuals →", out)

if __name__ == "__main__":
    main()
