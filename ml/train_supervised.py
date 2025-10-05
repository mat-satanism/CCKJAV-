import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from .mapping import normalize_labels, LABEL_CANDS, add_model_features

BASE_FEATURES = [
    "koi_period_log","koi_duration","koi_depth_log","koi_prad_log",
    "koi_steff","koi_slogg","koi_srad_log","koi_kepmag",
    "koi_depth_missing","koi_prad_missing","koi_srad_missing",
    "koi_steff_missing","koi_slogg_missing",
]

def pick_label_col(df: pd.DataFrame, user: str | None):
    if user:
        if user not in df.columns:
            raise KeyError(f"Label column '{user}' not found")
        return user
    for c in LABEL_CANDS:
        if c in df.columns:
            return c
    raise KeyError("No label column found (try --label-col koi_disposition / tfopwg_disp)")

def main():
    ap = argparse.ArgumentParser(description="Train supervised classifier (KOI)")
    ap.add_argument("--csv", required=True, help="data/cleaned/KOI_clean_for_learning.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--balance", choices=["none","class","smote"], default="class")
    ap.add_argument("--model", choices=["rf","logreg","xgb"], default="rf")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    df = add_model_features(df)

    ycol = pick_label_col(df, args.label_col)
    y_raw = normalize_labels(df[ycol])

    mask = y_raw.isin(["PLANET","FALSE POSITIVE"])
    X = df.loc[mask, BASE_FEATURES].astype(float).copy()
    y_raw = y_raw.loc[mask].reset_index(drop=True)

    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(0.0)

    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if args.balance == "class":
        class_weight = "balanced"
        smote = None
    elif args.balance == "smote":
        class_weight = None
        smote = SMOTE(random_state=42)
        Xtr, ytr = smote.fit_resample(Xtr, ytr)
    else:
        class_weight = None
        smote = None

    if args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=400, random_state=42, n_jobs=-1, class_weight=class_weight
        )
    elif args.model == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight=class_weight)
    else:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=42
        )

    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)
    roc = None
    try:
        roc = roc_auc_score(yte, proba[:, 1])
    except Exception:
        pass
    report = classification_report(yte, clf.predict(Xte), output_dict=True)

    dump(clf, out_dir / "model.joblib")
    dump(le,  out_dir / "label_encoder.joblib")
    (out_dir / "features.json").write_text(json.dumps(BASE_FEATURES, indent=2), encoding="utf-8")
    met = {
        "roc_auc_ovr": roc,
        "classification_report": report,
        "classes": le.classes_.tolist(),
        "label_col": ycol,
        "binary": True,
        "balance": args.balance,
    }
    (out_dir / "metrics.json").write_text(json.dumps(met, indent=2), encoding="utf-8")

    print(f"[OK] Saved artifacts → {out_dir}")

if __name__ == "__main__":
    main()