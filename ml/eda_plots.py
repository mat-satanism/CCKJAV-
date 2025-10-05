from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW8 = [
    "koi_period","koi_duration","koi_depth","koi_prad",
    "koi_steff","koi_slogg","koi_srad","koi_kepmag"
]
LABEL_CANDS = ["label","koi_disposition","tfopwg_disp","pred_class"]


def pick_label(df: pd.DataFrame, user_col: str | None) -> str:
    if user_col:
        if user_col not in df.columns:
            raise KeyError(f"Label column '{user_col}' not found")
        return user_col
    for c in LABEL_CANDS:
        if c in df.columns:
            return c
    raise KeyError("No label column found (try --label-col koi_disposition / tfopwg_disp / pred_class)")


def normalize_labels(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.upper().str.strip()
    s = s.replace({"CONFIRMED":"PLANET"})
    not_planet = {"FALSE POSITIVE","FP","NOT-PLANET","NO PLANET"}
    s = s.where(~s.isin(not_planet), "NOT-PLANET")
    s = s.where(s == "PLANET", "NOT-PLANET")
    return s


def ensure_columns(df: pd.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def scatter_pr(df: pd.DataFrame, y: pd.Series, outdir: Path):
    fig = plt.figure(figsize=(7,5))
    cls = (y == "PLANET").astype(int)
    x = pd.to_numeric(df["koi_period"], errors="coerce")
    yv = pd.to_numeric(df["koi_prad"], errors="coerce")
    m = np.isfinite(x) & np.isfinite(yv)
    plt.scatter(x[m], yv[m], c=cls[m], alpha=0.6)
    plt.xscale("log");
    plt.xlabel("Orbital period, days")
    plt.ylabel("Planet radius, R_Earth")
    plt.title("P vs R_p (colored by class)")
    fig.tight_layout()
    fig.savefig(outdir/"portrait_p_vs_r.png", dpi=200)
    plt.close(fig)


def corr_heatmap(df: pd.DataFrame, outdir: Path):
    cols = []
    for c in ["koi_period","koi_prad","koi_insol","koi_insol_log","koi_steff"]:
        if c in df.columns:
            cols.append(c)
    if len(cols) < 3:
        return
    M = df[cols].apply(pd.to_numeric, errors="coerce").corr()
    fig = plt.figure(figsize=(5,4))
    im = plt.imshow(M.values, interpolation="nearest")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlation heatmap")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir/"corr_heatmap.png", dpi=200)
    plt.close(fig)


def boxplots_by_class(df: pd.DataFrame, y: pd.Series, outdir: Path):
    metrics = [
        ("koi_period", "Orbital period (days)", True),
        ("koi_prad",   "Planet radius (R_Earth)", True),
        ("koi_steff",  "Stellar Teff (K)", False),
    ]
    for col, label, logx in metrics:
        if col not in df.columns:
            continue
        g1 = pd.to_numeric(df.loc[y=="PLANET", col], errors="coerce")
        g2 = pd.to_numeric(df.loc[y=="NOT-PLANET", col], errors="coerce")
        data = [g1[np.isfinite(g1)], g2[np.isfinite(g2)]]
        fig = plt.figure(figsize=(5,4))
        plt.boxplot(data, labels=["PLANET","NOT-PLANET"], showfliers=False)
        if logx:
            plt.yscale("log")
        plt.ylabel(label)
        plt.title(f"{col} by class")
        fig.tight_layout()
        fig.savefig(outdir/f"box_{col}.png", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="EDA: Portrait, Correlation Heatmap, Class-wise Boxplots")
    ap.add_argument("--csv", required=True, help="Input CSV with RAW8 and label column")
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    ensure_columns(df, ["koi_period","koi_prad"])  
    ycol = pick_label(df, args.label_col)
    y = normalize_labels(df[ycol])

    scatter_pr(df, y, outdir)
    corr_heatmap(df, outdir)
    boxplots_by_class(df, y, outdir)

    print("Saved EDA figures →", outdir)

if __name__ == "__main__":
    main()