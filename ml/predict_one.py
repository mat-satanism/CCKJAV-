from __future__ import annotations
import json, sys
import numpy as np
import pandas as pd
from pathlib import Path

from .model_utils import load_artifacts
from .mapping import RAW8, add_model_features

def predict_one(params: dict, artifacts_dir: str | Path = "artifacts"):
    base = Path(__file__).resolve().parents[1]
    model, le, features, metrics = load_artifacts(base / artifacts_dir)

    raw = pd.DataFrame([{k: params.get(k, np.nan) for k in RAW8}])
    # трансформуємо у MODEL-фічі
    X, feat_df = _prep_X(raw, features)
    proba = model.predict_proba(X.values)[0]
    classes = list(le.classes_)
    pred_idx = int(np.argmax(proba))
    return {
        "pred_class": str(le.inverse_transform([pred_idx])[0]),
        "probs": {classes[i]: float(proba[i]) for i in range(len(classes))},
        "features_used": features,
        "metrics": metrics,
        "transformed_row": feat_df.iloc[0].to_dict(),
    }

def _prep_X(raw_df: pd.DataFrame, features: list[str]):
    from .mapping import add_model_features
    df = add_model_features(raw_df)
    X = df.reindex(columns=features)
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(0.0)
    return X, df

if __name__ == "__main__":
    # CLI: python -m ml.predict_one '{"koi_period":10,"koi_duration":2,...}'
    if len(sys.argv) < 2:
        print("Usage: python -m ml.predict_one '{JSON}'")
        sys.exit(1)
    params = json.loads(sys.argv[1])
    res = predict_one(params)
    print(json.dumps(res, indent=2))