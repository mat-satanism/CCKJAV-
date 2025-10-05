# ml/toi_to_raw.py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from smart_read_csv import smart_read_csv

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

def _pick(df: pd.DataFrame, cands: list[str]):
    cols_norm = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    tried = []
    for cand in cands:
        key = cand.lower().replace(" ", "").replace("_", "")
        tried.append(key)
        if key in cols_norm:
            return cols_norm[key], tried
        key2 = key.replace("(days)", "").replace("(hours)", "").replace("(ppm)", "").replace("(r_earth)", "")
        if key2 in cols_norm:
            return cols_norm[key2], tried
    return None, tried

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', dest='out', required=True)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    df = smart_read_csv(args.inp)

    if args.debug:
        print("[DEBUG] Detected columns:", list(df.columns)[:40])

    out = pd.DataFrame()
    report = {}

    for k, cand in ALIASES.items():
        col, tried = _pick(df, cand)
        if col is None:
            avail = list(df.columns)[:20]
            msg = (f"Column for {k} not found.\n"
                   f"  Tried (normalized): {tried}\n"
                   f"  Available (first 20): {avail}\n"
                   f"  Tip: NASA-CSV files may have comments starting with '#'. We ignore them, but if the file was modified, check the delimiter and header.")
            raise ValueError(msg)
        report[k] = col

    out['koi_period']   = pd.to_numeric(df[report['koi_period']], errors='coerce')
    out['koi_duration'] = pd.to_numeric(df[report['koi_duration']], errors='coerce')
    out['koi_depth']    = pd.to_numeric(df[report['koi_depth']], errors='coerce')
    out['koi_prad']     = pd.to_numeric(df[report['koi_prad']], errors='coerce')
    out['koi_steff']    = pd.to_numeric(df[report['koi_steff']], errors='coerce')
    out['koi_slogg']    = pd.to_numeric(df[report['koi_slogg']], errors='coerce')
    out['koi_srad']     = pd.to_numeric(df[report['koi_srad']], errors='coerce')
    out['koi_kepmag']   = pd.to_numeric(df[report['koi_kepmag']], errors='coerce')

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=RAW_FEATURES)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print('[DONE] wrote RAW8 →', Path(args.out).resolve())
    print('[MAP] Column mapping used:')
    for k, v in report.items():
        print(f'  {k:14s} <- {v}')

if __name__ == '__main__':
    main()
