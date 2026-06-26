import os
import glob
import numpy as np
import pandas as pd

from . import config as C

def find_file(base):
    for cand in (base + ".csv", base):
        p = os.path.join(C.CERT_DIR, cand)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Nema datoteke '{base}' u {C.CERT_DIR}")

def header_of(base):
    return list(pd.read_csv(find_file(base), nrows=0).columns)

def pick_cols(base, wanted):
    have = header_of(base)
    return [c for c in wanted if c in have]

def count_rows(base):
    total = 0
    for ch in pd.read_csv(find_file(base), usecols=[0], chunksize=C.CHUNK_SIZE):
        total += len(ch)
    return total

def explore_dataset(verbose=True):
    summary = {}
    files = ["logon", "device", "http", "email", "file"]
    for base in files:
        path = find_file(base)
        size_mb = os.path.getsize(path) / 1e6
        head = pd.read_csv(path, nrows=1000)
        info = {
            "path": path,
            "size_mb": round(size_mb, 1),
            "columns": list(head.columns),
            "dtypes": {c: str(t) for c, t in head.dtypes.items()},
            "sample": head.head(3),
            "n_missing_in_sample": head.isna().sum().to_dict(),
        }
        summary[base] = info
        if verbose:
            print(f"\n=== {base.upper()} ===")
            print(f"  putanja:   {path}")
            print(f"  velicina:  {size_mb:.1f} MB")
            print(f"  stupci:    {list(head.columns)}")
            print(f"  tipovi:    {info['dtypes']}")
            print(f"  praznih (u uzorku od 1000): {info['n_missing_in_sample']}")
    return summary

def load_insiders():
    hits = glob.glob(os.path.join(C.ANSWERS_DIR, "**", "insiders.csv"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"Nema insiders.csv u {C.ANSWERS_DIR}")
    ins = pd.read_csv(hits[0])
    ins.columns = [c.strip().lower() for c in ins.columns]
    if "dataset" in ins.columns:
        ins = ins[ins["dataset"].astype(str).str.contains("4.2")]
    for col in ("start", "end"):
        d = pd.to_datetime(ins[col], format=C.DATE_FMT, errors="coerce")
        if d.isna().mean() > 0.5:
            d = pd.to_datetime(ins[col], errors="coerce")
        ins[col] = d
    ins = ins.dropna(subset=["start", "end"])
    return ins[["user", "start", "end"]].reset_index(drop=True)

def class_distribution(daily_df):
    n = len(daily_df)
    pos = int(daily_df["malicious"].sum())
    return {
        "ukupno_uzoraka": n,
        "zlonamjernih": pos,
        "normalnih": n - pos,
        "udio_zlonamjernih_%": round(100 * pos / n, 3),
        "omjer_neg_poz": round((n - pos) / max(pos, 1), 1),
    }

if __name__ == "__main__":
    explore_dataset()
    ins = load_insiders()
    print(f"\nInsajdera (r4.2): {ins['user'].nunique()}, prozora aktivnosti: {len(ins)}")
