import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np

def load_cluster_map(artifacts_ae: Path) -> dict:
    rep_path = artifacts_ae / "eval_report.json"
    if not rep_path.exists():
        return {}
    try:
        rep = json.load(open(rep_path, "r", encoding="utf-8"))
        clmap_raw = rep.get("cluster_label_map", {})
        clmap = {int(k): v for k, v in clmap_raw.items()}
        return clmap
    except Exception:
        return {}

def main():
    ap = argparse.ArgumentParser(description="Analyze AE+KMeans cluster outputs")
    ap.add_argument("--preds", default="preds/toi_ae_clusters.csv",
                    help="CSV with at least columns: cluster, cluster_dist (and preferably raw/features)")
    ap.add_argument("--artifacts_ae", default="artifacts_ae",
                    help="Folder with eval_report.json (optional)")
    ap.add_argument("--outdir", default="reports", help="Where to write reports")
    ap.add_argument("--topn", type=int, default=10, help="How many typical/outlier per cluster to save")
    args = ap.parse_args()

    preds_path = Path(args.preds).resolve()
    artifacts_ae = Path(args.artifacts_ae).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not preds_path.exists():
        raise FileNotFoundError(f"Cluster file not found: {preds_path}")

    df = pd.read_csv(preds_path)
    needed = {"cluster", "cluster_dist"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {preds_path.name}: {missing}")
    
    id_cols = [c for c in ["toi", "tid", "tfopwg_disp", "kepoi_name", "koi_disposition"] if c in df.columns]

    clmap = load_cluster_map(artifacts_ae)
    if clmap:
        df["cluster_majority"] = df["cluster"].map(clmap)
    else:
        df["cluster_majority"] = np.nan 

    df["cluster_score"] = 1.0 / (1.0 + df["cluster_dist"])

    agg_cols = {}
    if "tfopwg_disp" in df.columns:
        tf_counts = df.pivot_table(index="cluster", columns="tfopwg_disp",
                                   values=df.columns[0], aggfunc="count", fill_value=0)
        tf_counts = tf_counts.add_prefix("cnt_").reset_index()
    else:
        tf_counts = None

    base_stats = (
        df.groupby("cluster", as_index=False)
          .agg(
              n=("cluster", "size"),
              dist_mean=("cluster_dist", "mean"),
              dist_median=("cluster_dist", "median"),
              score_mean=("cluster_score", "mean"),
          )
          .sort_values(["n"], ascending=False)
    )

    ban = {"cluster", "cluster_dist", "cluster_score"}
    num_cols = [c for c in df.columns if c not in ban and pd.api.types.is_numeric_dtype(df[c])]
    cluster_profiles = df.groupby("cluster")[num_cols].mean().reset_index()

    topn = max(1, args.topn)
    top_typical = (
        df.sort_values(["cluster", "cluster_dist"], ascending=[True, True])
          .groupby("cluster", as_index=False)
          .head(topn)
          .reset_index(drop=True)
    )
    top_outliers = (
        df.sort_values(["cluster", "cluster_dist"], ascending=[True, False])
          .groupby("cluster", as_index=False)
          .head(topn)
          .reset_index(drop=True)
    )

    base_stats.to_csv(outdir / "cluster_summary.csv", index=False)
    cluster_profiles.to_csv(outdir / "cluster_profiles.csv", index=False)
    top_typical.to_csv(outdir / "top_typical.csv", index=False)
    top_outliers.to_csv(outdir / "top_outliers.csv", index=False)
    if tf_counts is not None:
        tf_counts.to_csv(outdir / "cluster_tfopwg_counts.csv", index=False)

    md_lines = [
        "# AE+KMeans Cluster Analysis",
        "",
        f"- Input: `{preds_path}`",
        f"- Artifacts: `{artifacts_ae}`",
        f"- Output tables: `{outdir}`",
        "",
        "## Files",
        "- `cluster_summary.csv` — cluster size, mean distance/score.",
        "- `cluster_profiles.csv` — mean values of numeric features per cluster.",
        "- `top_typical.csv` — Top-N most 'typical' objects in each cluster (min. distance).",
        "- `top_outliers.csv` — Top-N most 'anomalous' in each cluster (max. distance).",
    ]
    if tf_counts is not None:
        md_lines.append("- `cluster_tfopwg_counts.csv` — TFOPWG disposition frequencies per cluster.")
    if clmap:
        md_lines.append("- Used `cluster_label_map` from eval_report.json (majority KOI labels).")

    (outdir / "README.md").write_text("\n".join(md_lines), encoding="utf-8")

    print("Saved:")
    for f in ["cluster_summary.csv", "cluster_profiles.csv", "top_typical.csv", "top_outliers.csv", "README.md"]:
        print("   ", outdir / f)
    if tf_counts is not None:
        print("   ", outdir / "cluster_tfopwg_counts.csv")


if __name__ == "__main__":
    main()
