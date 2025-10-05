from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

def main():
    ap = argparse.ArgumentParser(description="Evaluate model predictions")
    ap.add_argument("--preds", required=True, help="CSV з колонками proba_planet, pred_class")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--label-col", help="Назва колонки з істинними класами (опціонально)")
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(df["proba_planet"].dropna().values, bins=30, color="skyblue", edgecolor="k")
    plt.xlabel("proba_planet")
    plt.ylabel("count")
    plt.title("Distribution of PLANET probabilities")
    plt.tight_layout()
    plt.savefig(out / "hist_proba.png", dpi=180)
    plt.close()

    # Orbital portrait
    if {"pl_orbper", "pl_rade"}.issubset(df.columns):
        plt.figure(figsize=(6, 4))
        plt.scatter(df["pl_orbper"], df["pl_rade"],
                    c=(df["pred_class"] == "PLANET").astype(int),
                    cmap="coolwarm", alpha=0.6)
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Orbital period (days)")
        plt.ylabel("Planet radius (R_Earth)")
        plt.title("P vs R_p (colored by prediction)")
        plt.tight_layout()
        plt.savefig(out / "portrait_P_vs_R.png", dpi=180)
        plt.close()

    # Heatmap correlations
    corr_cols = [c for c in ["pl_orbper","pl_trandurh","pl_trandep",
                             "pl_rade","pl_insol","st_teff","st_logg","st_rad"]
                 if c in df.columns]
    if len(corr_cols) >= 3:
        M = df[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
        fig = plt.figure(figsize=(5, 4))
        im = plt.imshow(M, cmap="coolwarm", interpolation="nearest")
        plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
        plt.yticks(range(len(corr_cols)), corr_cols)
        plt.title("Correlation heatmap")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out / "corr_heatmap.png", dpi=180)
        plt.close()

    # Classification report (if labels exist)
    if args.label_col and args.label_col in df.columns:
        y_true = df[args.label_col].astype(str).str.upper()
        y_pred = df["pred_class"].astype(str).str.upper()

        report = classification_report(y_true, y_pred, output_dict=True)
        with open(out / "classification_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(set(y_true)))
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(out / "confusion_matrix.png", dpi=180)
        plt.close()

        print("Saved metrics and plots to", out)
    else:
        print("No label column provided; skipping metrics.")

if __name__ == "__main__":
    main()
