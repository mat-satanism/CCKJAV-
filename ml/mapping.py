# ml/mapping.py
from __future__ import annotations
import numpy as np
import pandas as pd

RAW8 = [
    "koi_period",
    "koi_duration", 
    "koi_depth", 
    "koi_prad",
    "koi_steff", 
    "koi_slogg",  
    "koi_srad", 
    "koi_kepmag",
]

ALIASES = {
    "koi_period":   ["koi_period","pl_orbper","Period","Orbital Period","Period (days)"],
    "koi_duration": ["koi_duration","pl_trandurh","Transit Duration","Duration","tran_dur","tr_duration (hr)","trdur (hr)"],
    "koi_depth":    ["koi_depth","pl_trandep","Transit Depth","Depth","tr_depth (ppm)","depth_ppm"],
    "koi_prad":     ["koi_prad","pl_rade","Planet Radius","Rp","rp_rearth"],
    "koi_steff":    ["koi_steff","st_teff","Teff","T_eff"],
    "koi_slogg":    ["koi_slogg","st_logg","logg"],
    "koi_srad":     ["koi_srad","st_rad","Rstar","R_star (R_Sun)"],
    "koi_kepmag":   ["koi_kepmag","st_tmag","Tmag","KepMag"],
}

LABEL_CANDS = ["label","koi_disposition","tfopwg_disp","disp_3class","pred_class"]

def _norm(s: str) -> str:
    return (
        s.lower().replace(" ","").replace("_","")
         .replace("(days)","").replace("(hours)","")
         .replace("(ppm)","").replace("(r_sun)","").replace("(r_earth)","")
    )

def _pick(df: pd.DataFrame, opts: list[str]) -> str | None:
    lut = { _norm(c): c for c in df.columns }
    for cand in opts:
        key = _norm(cand)
        if key in lut:
            return lut[key]
    return None

def toi_to_koi_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for k, opts in ALIASES.items():
        col = _pick(df, opts)
        if col is None:
            out[k] = np.nan
        else:
            out[k] = pd.to_numeric(df[col], errors="coerce")
    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    return out

def add_model_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    for c in ["koi_depth","koi_prad","koi_srad","koi_steff","koi_slogg"]:
        df[f"{c}_missing"] = df[c].isna().astype(int)

    df["koi_period_log"] = np.log1p(pd.to_numeric(df["koi_period"], errors="coerce").clip(lower=0))
    df["koi_depth_log"]  = np.log1p(pd.to_numeric(df["koi_depth"],  errors="coerce").clip(lower=0))
    df["koi_prad_log"]   = np.log1p(pd.to_numeric(df["koi_prad"],   errors="coerce").clip(lower=0))
    df["koi_srad_log"]   = np.log1p(pd.to_numeric(df["koi_srad"],   errors="coerce").clip(lower=0))

    if "koi_insol" in raw_df.columns:
        v = pd.to_numeric(raw_df["koi_insol"], errors="coerce")
        df["koi_insol_log"]     = np.log1p(v.clip(lower=0))
        df["koi_insol_missing"] = v.isna().astype(int)

    return df

def normalize_labels(s: pd.Series) -> pd.Series:
    x = s.astype("string")
    x = x.str.strip().str.upper()

    x = x.replace({
        "CONFIRMED": "PLANET",
        "CP": "PLANET",
        "PC": "CANDIDATE",
        "FP": "FALSE POSITIVE",
        "FALSEPOSITIVE": "FALSE POSITIVE",
        "FALSE POSITIVES": "FALSE POSITIVE",
        "FALSE-POSITIVE": "FALSE POSITIVE",
    })

    x = x.fillna("CANDIDATE")
    return x
