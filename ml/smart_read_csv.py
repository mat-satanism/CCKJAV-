import csv
from pathlib import Path
import pandas as pd

def smart_read_csv(path: str | Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, comment="#", engine="python")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    try:
        with open(path, "rb") as fb:
            head = fb.read(50_000)
        head_txt = head.decode("utf-8", errors="ignore")
        try:
            dialect = csv.Sniffer().sniff(head_txt)
            delim = dialect.delimiter
        except Exception:
            delim = "\t" if head_txt.count("\t") > head_txt.count(",") else ","

        df = pd.read_csv(path, sep=delim, comment="#", engine="python")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    header_idx = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("loc_rowid,toi,tid") or (s.count(",") >= 10 and "toi" in s and "pl_orbper" in s):
            header_idx = i
            break

    if header_idx is not None:
        content = "".join(lines[header_idx:])
        from io import StringIO
        df = pd.read_csv(StringIO(content), engine="python")
        return df
    try:
        return pd.read_csv(path, sep=",", engine="python")
    except Exception:
        return pd.read_csv(path, sep="\t", engine="python")
