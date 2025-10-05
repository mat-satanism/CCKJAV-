from __future__ import annotations
import pandas as pd
import numpy as np

BASE_MODEL_FEATURES = [
    "koi_period_log","koi_duration","koi_depth_log","koi_prad_log",
    "koi_steff","koi_slogg","koi_srad_log","koi_kepmag",
    "koi_depth_missing","koi_prad_missing","koi_srad_missing",
    "koi_steff_missing","koi_slogg_missing",
]

def add_model_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Побудова похідних фіч: log1p, missing-флаги, опційно insolation/eqt."""
    df = raw.copy()
    # missing flags
    for c in ["koi_depth","koi_prad","koi_srad","koi_steff","koi_slogg"]:
        df[f"{c}_missing"] = df[c].isna().astype(int)
    # лог-версії
    for src, dst in [("koi_period","koi_period_log"),
                     ("koi_depth","koi_depth_log"),
                     ("koi_prad","koi_prad_log"),
                     ("koi_srad","koi_srad_log")]:
        df[dst] = np.log1p(pd.to_numeric(df[src], errors="coerce").clip(lower=0))
    # опційно: інсоляція/eqt
    if "koi_insol" in df.columns:
        df["koi_insol_log"] = np.log1p(pd.to_numeric(df["koi_insol"], errors="coerce").clip(lower=0))
        df["koi_insol_missing"] = df["koi_insol"].isna().astype(int)
    if "koi_eqt" in df.columns:
        df["koi_eqt_missing"] = df["koi_eqt"].isna().astype(int)
    return df


def pick_feature_list(df: pd.DataFrame) -> list[str]:
    feats = BASE_MODEL_FEATURES.copy()
    if "koi_insol_log" in df.columns:
        feats += ["koi_insol_log","koi_insol_missing"]
    if "koi_eqt" in df.columns:
        feats += ["koi_eqt","koi_eqt_missing"]
    return feats
