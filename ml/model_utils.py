import json
from pathlib import Path
from joblib import load
import pandas as pd
import numpy as np
from .mapping import add_model_features, toi_to_koi_raw

def load_artifacts(art_dir: Path):
    model = load(art_dir / "model.joblib")
    label_enc = load(art_dir / "label_encoder.joblib")
    features = json.loads((art_dir / "features.json").read_text())
    metrics  = json.loads((art_dir / "metrics.json").read_text())
    return model, label_enc, features, metrics

def ensure_features(df_in: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    raw = toi_to_koi_raw(df_in)
    model_df = add_model_features(raw)
    X = model_df.reindex(columns=features)
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(0.0)
    return X, model_df