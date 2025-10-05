from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from joblib import dump

# локальні модулі
from ml.mapping import normalize_labels, LABEL_CANDS
from ml.features import add_model_features, pick_feature_list

def try_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric="logloss", n_jobs=-1, random_state=42
        )
    except Exception:
        return None

def get_base_models():
    models = {
        "logreg": LogisticRegression(max_iter=5000, class_weight="balanced", n_jobs=-1),
        "rf": RandomForestClassifier(n_estimators=600, class_weight="balanced", n_jobs=-1, random_state=42),
        "hgb": HistGradientBoostingClassifier(max_depth=None, learning_rate=0.08, max_bins=255, random_state=42),
    }
    xgb = try_xgb()
    if xgb is not None:
        models["xgb"] = xgb
    return models

def fit_and_calibrate(model, X, y):
    """Калібруємо ізотронікою поверх імовірностей (3-fold)."""
    calib = CalibratedClassifierCV(model, method="isotonic", cv=3)
    return calib.fit(X, y)

def main():
    ap = argparse.ArgumentParser(description="Train supervised models with CV, calibration, thresholding and stacking")
    ap.add_argument("--csv", required=True, help="KOI_clean_for_learning.csv (або інший KOI-датасет з мітками)")
    ap.add_argument("--out", default="artifacts", help="Куди зберігати артефакти")
    ap.add_argument("--balance", choices=["none","class","smote"], default="class")
    ap.add_argument("--stack", action="store_true", help="Увімкнути ансамбль (stacking)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    # вибираємо колонку мітки
    ycol = next((c for c in LABEL_CANDS if c in df.columns), None)
    if ycol is None:
        raise KeyError(f"Не знайдено label-колонки серед {LABEL_CANDS}")
    y_raw = normalize_labels(df[ycol])
    # бінарка: PLANET vs FALSE POSITIVE
    m = y_raw.isin(["PLANET","FALSE POSITIVE"])
    df = df.loc[m].reset_index(drop=True)
    y_raw = y_raw.loc[m].reset_index(drop=True)

    # фічі
    df = add_model_features(df)
    FEATURES = pick_feature_list(df)
    X = df[FEATURES].astype(float)
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(0.0)

    # лейбли
    le = LabelEncoder().fit(y_raw)
    y = (le.transform(y_raw) == list(le.classes_).index("PLANET")).astype(int)

    # (опц.) SMOTE
    if args.balance == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            X, y = SMOTE(random_state=42).fit_resample(X, y)
        except Exception as e:
            print("[WARN] SMOTE недоступний:", e)

    # CV оцінка та калібрування базових моделей
    M = get_base_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {}
    fitted_models = {}

    for name, est in M.items():
        oof_prob = np.zeros(len(y), dtype=float)
        print(f"\n[CV] {name}")
        for tr, te in cv.split(X, y):
            est_c = fit_and_calibrate(est, X.iloc[tr], y[tr])
            oof_prob[te] = est_c.predict_proba(X.iloc[te])[:,1]
        roc = roc_auc_score(y, oof_prob)
        pr  = average_precision_score(y, oof_prob)
        cv_scores[name] = {"roc_auc": float(roc), "pr_auc": float(pr)}
        # фінальне підгоняння на всіх даних + калібрування
        fitted_models[name] = fit_and_calibrate(est, X, y)
        print(f"  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")

    # Ансамбль (blending): середнє по каліброваних імовірностях
    blend_prob = None
    for name, mdl in fitted_models.items():
        p = mdl.predict_proba(X)[:,1]
        blend_prob = p if blend_prob is None else (blend_prob + p)
    blend_prob = blend_prob / len(fitted_models)

    # Поріг за F1
    p, r, thr = precision_recall_curve(y, blend_prob)
    f1 = (2*p*r) / np.clip(p+r, 1e-12, None)
    i = int(np.nanargmax(f1))
    best_thr = float(thr[max(i-1, 0)])
    print(f"\n[THRESH] best F1 threshold ≈ {best_thr:.3f}")

    # Підсумкова метрика на train (для референсу)
    y_pred = (blend_prob >= best_thr).astype(int)
    rep = classification_report(y, y_pred, target_names=["FALSE POSITIVE","PLANET"], output_dict=True)
    cm  = confusion_matrix(y, y_pred).tolist()

    # Збереження
    for name, mdl in fitted_models.items():
        dump(mdl, out / f"model_{name}.joblib")
    (out / "features.json").write_text(json.dumps(FEATURES, indent=2), encoding="utf-8")
    (out / "blend.json").write_text(json.dumps({
        "members": list(fitted_models.keys()),
        "threshold": best_thr,
        "cv_scores": cv_scores,
    }, indent=2), encoding="utf-8")
    dump(le, out / "label_encoder.joblib")

    (out / "metrics.json").write_text(json.dumps({
        "cv_scores": cv_scores,
        "train_report": rep,
        "confusion_matrix": cm,
        "threshold": best_thr,
        "classes": ["FALSE POSITIVE","PLANET"],
        "features_used": FEATURES
    }, indent=2), encoding="utf-8")

    print("\n✅ Saved artifacts →", out)

if __name__ == "__main__":
    main()
