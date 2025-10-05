# ml/mapping.py
from __future__ import annotations
import numpy as np
import pandas as pd

# 8 "сирих" ознак (вводяться у вкладці Single, або мапляться з TOI/KOI)
RAW8 = [
    "koi_period",   # days
    "koi_duration", # hours
    "koi_depth",    # ppm
    "koi_prad",     # R_Earth
    "koi_steff",    # K
    "koi_slogg",    # log g
    "koi_srad",     # R_Sun
    "koi_kepmag",   # TESS/Kep magnitude
]

# Найчастіші назви колонок у KOI/TOI CSV
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

# Можливі стовпці з мітками у KOI/TOI
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
    """
    Прочитує TOI/KOI CSV і повертає таблицю з колонками RAW8.
    Якщо чогось не вистачає — ставить NaN (не падає).
    """
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
    """
    З RAW8 робимо набір MODEL-фіч:
      - лог-фічі: log1p для period/depth/prad/srad
      - флаги пропусків *_missing
      - (опційно) інсоляція, якщо присутня у вхідному DF як 'koi_insol'
    """
    df = raw_df.copy()

    # flags
    for c in ["koi_depth","koi_prad","koi_srad","koi_steff","koi_slogg"]:
        df[f"{c}_missing"] = df[c].isna().astype(int)

    # logs (safe)
    df["koi_period_log"] = np.log1p(pd.to_numeric(df["koi_period"], errors="coerce").clip(lower=0))
    df["koi_depth_log"]  = np.log1p(pd.to_numeric(df["koi_depth"],  errors="coerce").clip(lower=0))
    df["koi_prad_log"]   = np.log1p(pd.to_numeric(df["koi_prad"],   errors="coerce").clip(lower=0))
    df["koi_srad_log"]   = np.log1p(pd.to_numeric(df["koi_srad"],   errors="coerce").clip(lower=0))

    # optional insolation if present (already KOI-like name)
    if "koi_insol" in raw_df.columns:
        v = pd.to_numeric(raw_df["koi_insol"], errors="coerce")
        df["koi_insol_log"]     = np.log1p(v.clip(lower=0))
        df["koi_insol_missing"] = v.isna().astype(int)

    return df

def normalize_labels(s: pd.Series) -> pd.Series:
    """
    Уніфікує позначення у PLANET / CANDIDATE / FALSE POSITIVE.
    Стійко обробляє NaN та змішані типи.
    """
    x = s.astype("string")  # pandas StringDtype, зручно для .str-операцій
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

    # інколи трапляються порожні / None → маркуємо як "CANDIDATE" за замовчуванням або залишаємо як є
    x = x.fillna("CANDIDATE")
    return x
