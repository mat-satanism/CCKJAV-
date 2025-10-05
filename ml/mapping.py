from __future__ import annotations
import pandas as pd
import numpy as np

RAW_FEATURES = [
    'koi_period','koi_duration','koi_depth','koi_prad',
    'koi_steff','koi_slogg','koi_srad','koi_kepmag'
]

ALIASES = {
    'koi_period':   ['pl_orbper','Period','Orbital Period','Period (days)'],
    'koi_duration': ['pl_trandurh','Transit Duration','Duration','tran_dur','tr_duration (hr)','trdur (hr)'],
    'koi_depth':    ['pl_trandep','Transit Depth','Depth','tr_depth (ppm)','depth_ppm'],
    'koi_prad':     ['pl_rade','Planet Radius','Rp','rp_rearth'],
    'koi_steff':    ['st_teff','Teff','T_eff'],
    'koi_slogg':    ['st_logg','logg'],
    'koi_srad':     ['st_rad','Rstar','R_star (R_Sun)'],
    'koi_kepmag':   ['st_tmag','Tmag','KepMag'],
}


def _normalize(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("(days)","")\
        .replace("(hours)","").replace("(ppm)","").replace("(r_earth)","")


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm2orig = { _normalize(c): c for c in df.columns }
    for cand in candidates:
        key = _normalize(cand)
        if key in norm2orig:
            return norm2orig[key]
    return None


def toi_to_koi_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for k, opts in ALIASES.items():
        col = pick_column(df, opts)
        if col is None:
            raise ValueError(f"Required column for {k} not found; tried {opts[:4]}...")
        out[k] = pd.to_numeric(df[col], errors='coerce')
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def add_model_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    for c in ['koi_depth', 'koi_prad', 'koi_srad', 'koi_steff', 'koi_slogg']:
        df[f"{c}_missing"] = df[c].isna().astype(int)
    df['koi_period_log'] = np.log1p(df['koi_period'].clip(lower=0))
    df['koi_depth_log']  = np.log1p(df['koi_depth'].clip(lower=0))
    df['koi_prad_log']   = np.log1p(df['koi_prad'].clip(lower=0))
    df['koi_srad_log']   = np.log1p(df['koi_srad'].clip(lower=0))
    return df