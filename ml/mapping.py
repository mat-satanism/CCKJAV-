from __future__ import annotations
import pandas as pd
import numpy as np

# KOI-узгоджені "сирі" фічі (RAW8)
RAW8 = [
    "koi_period","koi_duration","koi_depth","koi_prad",
    "koi_steff","koi_slogg","koi_srad","koi_kepmag"
]

# Поширені варіанти назв у KOI/TOI/TESS (враховано архівний TESS.csv і твій tess_cleaned)
ALIASES = {
    "koi_period":   ["koi_period","pl_orbper","Period","Orbital Period","orbper","period_days","log_orbper"],
    "koi_duration": ["koi_duration","pl_trandurh","Transit Duration","Duration","tran_dur","tr_duration (hr)","trdur (hr)","trdur","tdur","t_dur","duration_hr","duration_hrs","dur_hr","dur_hrs","dur (hr)","transit_dur","transit_duration"],
    "koi_depth":    ["koi_depth","pl_trandep","Transit Depth","Depth","tr_depth (ppm)","depth_ppm","trdepthppm","log_trandep","depth_norm"],
    "koi_prad":     ["koi_prad","pl_rade","Planet Radius","Rp","rp_rearth","radius_re","pl_rade_re","sqrt_pl_rade","rel_radius"],
    "koi_steff":    ["koi_steff","st_teff","Teff","T_eff","teff"],
    "koi_slogg":    ["koi_slogg","st_logg","logg","log_g","stlogg"],
    "koi_srad":     ["koi_srad","st_rad","Rstar","R_star (R_Sun)","rstar","st_radius"],
    "koi_kepmag":   ["koi_kepmag","st_tmag","Tmag","KepMag","kepmag","tmag"],
    # опції, якщо є
    "koi_insol":    ["koi_insol","pl_insol","insol","insolation"],
    "koi_eqt":      ["koi_eqt","pl_eqt","eqt","eq_temperature"],
}

LABEL_CANDS = ["label","koi_disposition","tfopwg_disp","disp_3class","pred_class"]


def _norm(s: str) -> str:
    s = s.lower()
    for ch in " _-()[]{}:/,":
        s = s.replace(ch, "")
    s = s.replace("hours","hr").replace("hour","hr").replace("hrs","hr").replace(" h","hr")
    s = s.replace("days","day").replace(" d","day")
    s = s.replace("ppm","")
    return s


def _pick_col(df: pd.DataFrame, names: list[str]) -> str|None:
    norm2orig = {_norm(c): c for c in df.columns}
    for cand in names:
        key = _norm(cand)
        if key in norm2orig:
            return norm2orig[key]
    # додаткова евристика для duration
    for c in df.columns:
        n = _norm(c)
        if ("dur" in n or "duration" in n or n.startswith("tdur") or n.startswith("trdur")) and ("hr" in n or "duration" in n or "dur" in n):
            return c
    return None


def toi_to_koi_raw(df_in: pd.DataFrame) -> pd.DataFrame:
    """Витягнути KOI-подібні RAW8 з будь-якої TOI/TESS/cleaned таблиці."""
    out = pd.DataFrame(index=df_in.index)
    missing = []
    for k in RAW8:
        col = _pick_col(df_in, ALIASES.get(k, [k]))
        if col is None:
            missing.append(k)
        else:
            out[k] = pd.to_numeric(df_in[col], errors="coerce")
    # не критично відсутні (опції)
    for k in ["koi_insol","koi_eqt"]:
        col = _pick_col(df_in, ALIASES.get(k, [k]))
        out[k] = pd.to_numeric(df_in[col], errors="coerce") if col else np.nan
    if missing:
        # дамо підказку користувачу
        avail = list(df_in.columns)[:50]
        raise KeyError(f"Не знайшов колонки для: {missing}. Перевір назви у вхідному CSV. Перші 50 колонок: {avail}")
    return out


def normalize_labels(y: pd.Series) -> pd.Series:
    """Нормалізувати KOI/TOI диспо до PLANET vs FALSE POSITIVE (PC/Candidate → ігноруємо у бінарці)."""
    s = y.astype(str).str.upper().str.strip()
    s = s.replace({"CONFIRMED":"PLANET","CP":"PLANET","KP":"PLANET","PC":"CANDIDATE","CANDIDATE":"CANDIDATE"})
    s = s.replace({"FP":"FALSE POSITIVE","FALSEPOSITIVE":"FALSE POSITIVE"})
    return s
