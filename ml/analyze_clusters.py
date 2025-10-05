from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Analyze AE+KMeans cluster outputs")
    ap.add_argument("--preds", default="preds/ae_clusters.csv")
    ap.add_argument("--artifacts_ae", default="artifacts_ae")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()

    preds_path = Path(args.preds).resolve()
    outdir = Path(args.outdir).resolve(); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(preds_path)
    if not {"cluster","cluster_dist"}.issubset(df.columns):
        raise ValueError("preds missing required cols: cluster, cluster_dist")

    df["cluster_score"] = 1.0 / (1.0 + df["cluster_dist"])  

    base_stats = (
        df.groupby("cluster", as_index=False)
          .agg(n=("cluster","size"), dist_mean=("cluster_dist","mean"),
               dist_median=("cluster_dist","median"), score_mean=("cluster_score","mean"))
          .sort_values("n", ascending=False)
    )

    ban = {"cluster","cluster_dist","cluster_score"}
    num_cols = [c for c in df.columns if c not in ban and pd.api.types.is_numeric_dtype(df[c])]
    cluster_profiles = df.groupby("cluster")[num_cols].mean().reset_index()

    topn = max(1, args.topn)
    top_typical = (df.sort_values(["cluster","cluster_dist"]).groupby("cluster").head(topn).reset_index(drop=True))
    top_outliers = (df.sort_values(["cluster","cluster_dist"], ascending=[True, False]).groupby("cluster").head(topn).reset_index(drop=True))

    base_stats.to_csv(outdir/"cluster_summary.csv", index=False)
    cluster_profiles.to_csv(outdir/"cluster_profiles.csv", index=False)
    top_typical.to_csv(outdir/"top_typical.csv", index=False)
    top_outliers.to_csv(outdir/"top_outliers.csv", index=False)

    md = ["# AE+KMeans Cluster Analysis", "", f"Input: `{preds_path}`", "", "Files:",
          "- cluster_summary.csv", "- cluster_profiles.csv", "- top_typical.csv", "- top_outliers.csv"]
    (outdir/"README.md").write_text("\n".join(md), encoding="utf-8")

    print("Saved reports →", outdir)


if __name__ == "__main__":
    main()