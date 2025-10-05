from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from joblib import dump

BASE_FEATURES_MODEL = [
    "koi_period_log", "koi_duration", "koi_depth_log", "koi_prad_log",
    "koi_steff", "koi_slogg", "koi_srad_log", "koi_kepmag",
    "koi_depth_missing", "koi_prad_missing", "koi_srad_missing",
    "koi_steff_missing", "koi_slogg_missing",
]

RAW_FEATURES = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
]

LABEL_CANDIDATES = ["label", "koi_disposition", "tfopwg_disp"]


def has_all(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


def pick_label_column(df: pd.DataFrame, user_col: str | None) -> str:
    if user_col:
        if user_col not in df.columns:
            raise KeyError(f"Label column '{user_col}' not found in CSV")
        return user_col
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError("No label column found. Try --label-col koi_disposition (KOI) or tfopwg_disp (TOI).")


def ensure_model_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feats = BASE_FEATURES_MODEL.copy()
    out = df.copy()

    if "koi_insol_log" in out.columns or "koi_insol" in out.columns:
        if "koi_insol_log" not in out.columns and "koi_insol" in out.columns:
            out["koi_insol_log"] = np.log1p(pd.to_numeric(out["koi_insol"], errors="coerce").clip(lower=0))
        out["koi_insol_missing"] = out.get("koi_insol").isna().astype(int) if "koi_insol" in out.columns else 0
        feats += ["koi_insol_log", "koi_insol_missing"]

    if has_all(out, feats):
        return out, feats

    if not has_all(out, RAW_FEATURES):
        missing = [c for c in RAW_FEATURES if c not in out.columns]
        raise KeyError(f"CSV lacks MODEL features and RAW8 is incomplete; missing: {missing}")

    from ml.mapping import add_model_features
    derived = add_model_features(out[RAW_FEATURES].copy())
    for c in BASE_FEATURES_MODEL:
        out[c] = derived[c]

    if "koi_insol" in df.columns and "koi_insol_log" not in out.columns:
        out["koi_insol_log"] = np.log1p(pd.to_numeric(df["koi_insol"], errors="coerce").clip(lower=0))
        out["koi_insol_missing"] = df["koi_insol"].isna().astype(int)
        feats += ["koi_insol_log", "koi_insol_missing"]

    return out, feats


def main():
    ap = argparse.ArgumentParser(description="Train supervised baseline model (RF or LogReg)")
    ap.add_argument("--csv", required=True, help="Input KOI/TOI CSV")
    ap.add_argument("--out", default="artifacts", help="Artifacts output dir")
    ap.add_argument("--model", choices=["rf", "logreg"], default="rf")
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--binary", choices=["auto", "cf", "all"], default="auto",
                    help="'cf' keeps CONFIRMED vs FALSE POSITIVE, 'all' keeps all")
    ap.add_argument("--balance", choices=["none", "class", "smote"], default="class",
                    help="'class' = class_weight='balanced', 'smote' = oversample minority")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    label_col = pick_label_column(df, args.label_col)
    y_raw = df[label_col].astype(str).str.upper().str.strip()
    if args.binary in ("auto", "cf"):
        mask = y_raw.isin(["CONFIRMED", "FALSE POSITIVE"])
        df = df.loc[mask].reset_index(drop=True)
        y_raw = y_raw.loc[mask].reset_index(drop=True).replace({"CONFIRMED": "PLANET"})

    df, FEATURES = ensure_model_features(df)
    X = df[FEATURES].astype(float)
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(0.0)

    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    class_weight = None
    if args.balance == "class":
        class_weight = "balanced"
    elif args.balance == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            Xtr, ytr = sm.fit_resample(Xtr, ytr)
        except Exception as e:
            print("[WARN] SMOTE requested but imblearn not available:", e)

    if args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=400, n_jobs=-1, random_state=42,
            class_weight=class_weight, min_samples_split=2
        )
    else:
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight=class_weight)

    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)
    try:
        roc_auc_ovr = roc_auc_score(yte, proba, multi_class="ovr")
    except Exception:
        roc_auc_ovr = None
    y_pred = clf.predict(Xte)
    report = classification_report(yte, y_pred, output_dict=True)

    dump(clf, out_dir / "model.joblib")
    dump(le, out_dir / "label_encoder.joblib")
    (out_dir / "features.json").write_text(json.dumps(FEATURES, indent=2), encoding="utf-8")

    metrics = {
        "roc_auc_ovr": roc_auc_ovr,
        "classification_report": report,
        "n_classes": len(le.classes_),
        "classes": le.classes_.tolist(),
        "label_col": label_col,
        "binary": args.binary,
        "balance": args.balance,
        "features_used": FEATURES,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[Supervised] Модель '{args.model}' навчено та збережено у → {out_dir}")
    print(f"ROC-AUC (ovr): {roc_auc_ovr}")
    print("Класи:", le.classes_.tolist())


if __name__ == "__main__":
    main()