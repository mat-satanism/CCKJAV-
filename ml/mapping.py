# ml/mapping.py
from __future__ import annotations
import pandas as pd
import numpy as np
from ml.mapping import RAW8, toi_to_koi_raw, add_model_features, LABEL_CANDS, normalize_labels


# 8 "сирих" ознак, які користувач вводить у першій вкладці
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

# можливі назви у KOI/TOI (Exoplanet Archive)
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
        s.lower()
         .replace(" ", "")
         .replace("_","")
         .replace("(days)","")
         .replace("(hours)","")
         .replace("(ppm)","")
         .replace("(r_sun)","")
         .replace("(r_earth)","")
    )

def _pick(df: pd.DataFrame, opts: list[str]) -> str | None:
    lut = { _norm(c): c for c in df.columns }
    for cand in opts:
        key = _norm(cand)
        if key in lut:
            return lut[key]
    return None

def toi_to_koi_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Вибирає/перейменовує колонки з TOI/KOI у єдині RAW8."""
    out = pd.DataFrame(index=df.index)
    missing = []
    for k, opts in ALIASES.items():
        col = _pick(df, opts)
        if col is None:
            missing.append(k)
            out[k] = np.nan
        else:
            out[k] = pd.to_numeric(df[col], errors="coerce")
    if missing:
        # не падаємо, але корисно знати
        # raise KeyError(f"Не знайшов колонки для: {missing}. Перевір файл.")
        pass
    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    return out

def add_model_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """З RAW8 робимо повний набір для моделі (лог-фічі + flags пропусків)."""
    df = raw_df.copy()
    # флаги пропусків
    for c in ["koi_depth","koi_prad","koi_srad","koi_steff","koi_slogg"]:
        df[f"{c}_missing"] = df[c].isna().astype(int)

    # лог-фічі (безпечні)
    df["koi_period_log"] = np.log1p(df["koi_period"].clip(lower=0))
    df["koi_depth_log"]  = np.log1p(df["koi_depth"].clip(lower=0))
    df["koi_prad_log"]   = np.log1p(df["koi_prad"].clip(lower=0))
    df["koi_srad_log"]   = np.log1p(df["koi_srad"].clip(lower=0))

    # якщо є інсоляція — теж додамо
    if "koi_insol" in raw_df.columns:
        df["koi_insol_log"]     = np.log1p(pd.to_numeric(raw_df["koi_insol"], errors="coerce").clip(lower=0))
        df["koi_insol_missing"] = raw_df["koi_insol"].isna().astype(int)

    return df

def normalize_labels(s: pd.Series) -> pd.Series:
    """Нормалізує мітки KOI/TOI у PLANET / CANDIDATE / FALSE POSITIVE."""
    x = s.astype(str).str.upper().str.strip()
    mapping = {
        "CONFIRMED": "PLANET",
        "CP": "PLANET",
        "PC": "CANDIDATE",
        "FP": "FALSE POSITIVE",
        "FALSEPOSITIVE": "FALSE POSITIVE",
    }
    x = x.replace(mapping)
    return x
