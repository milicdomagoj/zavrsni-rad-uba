import os
import glob
import pandas as pd
import numpy as np

CERT_DIR = r"C:\Users\Domagoj\Desktop\r4.2"
ANSWERS_DIR = r"C:\Users\Domagoj\Desktop\answers"
CHUNK_SIZE = 2_000_000
WORK_START, WORK_END = 8, 18

CERT_INTERNAL_DOMAIN = "dtaa.com"
CERT_SUSPICIOUS_SITES = ["wikileaks", "job-hunt", "monster", "careerbuilder",
                         "indeed", "simplyhired", "keylog"]

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cert_cache")
FEATURES_CACHE = os.path.join(CACHE_DIR, "features_full.csv")
DATE_FMT = "%m/%d/%Y %H:%M:%S"
DAY_FMT = "%m/%d/%Y"

FEATURE_COLS = ["num_logons", "num_after_hours_logons", "num_weekend_logons",
                "num_distinct_pcs", "num_usb_connects", "num_after_hours_usb",
                "num_files", "num_files_to_usb", "num_emails",
                "num_external_emails", "total_email_size", "num_attachments",
                "num_http", "num_suspicious_http"]

def _find_file(base):
    for cand in (base + ".csv", base):
        path = os.path.join(CERT_DIR, cand)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Ne mogu naci datoteku '{base}' u {CERT_DIR}")

def _read_header(base):
    return list(pd.read_csv(_find_file(base), nrows=0).columns)

def _pick(base, wanted):
    cols = _read_header(base)
    return [c for c in wanted if c in cols]

def _logon_features():
    path = _find_file("logon")
    df = pd.read_csv(path, usecols=_pick("logon", ["date", "user", "pc", "activity"]))
    df = df[df["activity"] == "Logon"].copy()
    dt = pd.to_datetime(df["date"], format=DATE_FMT, errors="coerce")
    df = df[dt.notna()].copy()
    dt = dt[dt.notna()]
    df["day"] = dt.dt.date
    df["after"] = ((dt.dt.hour < WORK_START) | (dt.dt.hour >= WORK_END)).astype(int)
    df["weekend"] = (dt.dt.weekday >= 5).astype(int)
    g = df.groupby(["user", "day"]).agg(
        num_logons=("activity", "size"),
        num_after_hours_logons=("after", "sum"),
        num_weekend_logons=("weekend", "sum"),
        num_distinct_pcs=("pc", "nunique"),
    ).reset_index()
    return g

def _device_features():
    path = _find_file("device")
    df = pd.read_csv(path, usecols=_pick("device", ["date", "user", "pc", "activity"]))
    df = df[df["activity"] == "Connect"].copy()
    dt = pd.to_datetime(df["date"], format=DATE_FMT, errors="coerce")
    df = df[dt.notna()].copy()
    dt = dt[dt.notna()]
    df["day"] = dt.dt.date
    df["after"] = ((dt.dt.hour < WORK_START) | (dt.dt.hour >= WORK_END)).astype(int)
    g = df.groupby(["user", "day"]).agg(
        num_usb_connects=("activity", "size"),
        num_after_hours_usb=("after", "sum"),
    ).reset_index()
    return g

def _file_features():
    path = _find_file("file")
    df = pd.read_csv(path, usecols=_pick("file", ["date", "user", "pc", "filename"]))
    df["day"] = pd.to_datetime(df["date"], format=DATE_FMT, errors="coerce").dt.date
    df = df.dropna(subset=["day"])
    g = df.groupby(["user", "day"]).agg(num_files=("filename", "size")).reset_index()
    g["num_files_to_usb"] = g["num_files"]
    return g

def _day_from_str(series):
    return series.str.slice(0, 10)

def _email_features():
    cols = _pick("email", ["date", "user", "to", "cc", "bcc", "size", "attachments"])
    path = _find_file("email")
    parts = []
    total = 0
    for chunk in pd.read_csv(path, usecols=cols, chunksize=CHUNK_SIZE):
        total += len(chunk)
        chunk["day"] = _day_from_str(chunk["date"].astype(str))
        recips = chunk["to"].fillna("").astype(str)
        for c in ("cc", "bcc"):
            if c in chunk.columns:
                recips = recips + ";" + chunk[c].fillna("").astype(str)
        chunk["external"] = (~recips.str.contains(CERT_INTERNAL_DOMAIN, na=False)).astype(int)
        chunk["size"] = pd.to_numeric(chunk.get("size", 0), errors="coerce").fillna(0)
        att = chunk.get("attachments", 0)
        chunk["attachments"] = pd.to_numeric(att, errors="coerce").fillna(0)
        g = chunk.groupby(["user", "day"]).agg(
            num_emails=("external", "size"),
            num_external_emails=("external", "sum"),
            total_email_size=("size", "sum"),
            num_attachments=("attachments", "sum"),
        ).reset_index()
        parts.append(g)
        print(f"      email: obradjeno {total:,} redaka...", end="\r")
    print(f"      email: gotovo ({total:,} redaka)                    ")
    allg = pd.concat(parts, ignore_index=True)
    return allg.groupby(["user", "day"], as_index=False).sum()

def _http_features():
    cols = _pick("http", ["date", "user", "url"])
    path = _find_file("http")
    pat = "|".join(CERT_SUSPICIOUS_SITES)
    parts = []
    total = 0
    for chunk in pd.read_csv(path, usecols=cols, chunksize=CHUNK_SIZE):
        total += len(chunk)
        chunk["day"] = _day_from_str(chunk["date"].astype(str))
        chunk["susp"] = chunk["url"].astype(str).str.contains(pat, case=False, na=False).astype(int)
        g = chunk.groupby(["user", "day"]).agg(
            num_http=("susp", "size"),
            num_suspicious_http=("susp", "sum"),
        ).reset_index()
        parts.append(g)
        print(f"      http: obradjeno {total:,} redaka...", end="\r")
    print(f"      http: gotovo ({total:,} redaka)                    ")
    allg = pd.concat(parts, ignore_index=True)
    return allg.groupby(["user", "day"], as_index=False).sum()

def load_insiders():
    hits = glob.glob(os.path.join(ANSWERS_DIR, "**", "insiders.csv"), recursive=True)
    if not hits:
        raise FileNotFoundError(
            f"Ne mogu naci insiders.csv u {ANSWERS_DIR}. Raspakiraj answers.tar.")
    ins = pd.read_csv(hits[0])
    ins.columns = [c.strip().lower() for c in ins.columns]
    if "dataset" in ins.columns:
        ins = ins[ins["dataset"].astype(str).str.contains("4.2")]

    def _parse(col):
        d = pd.to_datetime(ins[col], format=DATE_FMT, errors="coerce")
        if d.isna().mean() > 0.5:
            d = pd.to_datetime(ins[col], errors="coerce")
        return d

    ins["start"] = _parse("start")
    ins["end"] = _parse("end")
    ins = ins.dropna(subset=["start", "end"])
    return ins[["user", "start", "end"]].reset_index(drop=True)

def build_ground_truth(features_df):
    insiders = load_insiders()
    print(f"    Insajdera (r4.2) pronadjeno: {insiders['user'].nunique()}")

    malicious_pairs = set()
    for _, w in insiders.iterrows():
        days = pd.date_range(w["start"].normalize(), w["end"].normalize(), freq="D")
        for d in days:
            malicious_pairs.add((w["user"], d.date()))

    key = list(zip(features_df["user"], features_df["day"]))
    features_df["malicious"] = [1 if k in malicious_pairs else 0 for k in key]
    return features_df

def build_full_feature_matrix(use_cache=True):
    if use_cache and os.path.exists(FEATURES_CACHE):
        print(f"[cache] Ucitavam gotovu matricu znacajki: {FEATURES_CACHE}")
        df = pd.read_csv(FEATURES_CACHE)
        df["day"] = pd.to_datetime(df["day"]).dt.date
        return df, FEATURE_COLS

    print("[*] Racunanje znacajki iz malih datoteka (logon, device, file)...")
    logon = _logon_features()
    device = _device_features()
    file_df = _file_features()

    print("[*] Streaming znacajki iz velikih datoteka (email, http)...")
    email = _email_features()
    http = _http_features()

    print("[*] Spajanje znacajki...")
    for d in (email, http):
        d["day"] = pd.to_datetime(d["day"], format=DAY_FMT, errors="coerce").dt.date
        d.dropna(subset=["day"], inplace=True)

    df = logon
    for other in (device, file_df, email, http):
        df = df.merge(other, on=["user", "day"], how="outer")

    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    df = df[["user", "day"] + FEATURE_COLS].sort_values(["user", "day"]).reset_index(drop=True)

    print("[*] Pridruzivanje oznaka (ground truth)...")
    df = build_ground_truth(df)

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(FEATURES_CACHE, index=False)
    print(f"    Matrica spremljena u cache: {FEATURES_CACHE}")
    return df, FEATURE_COLS

if __name__ == "__main__":
    df, cols = build_full_feature_matrix()
    print(f"\nMatrica: {df.shape[0]} (user-dan) x {len(cols)} znacajki")
    print(f"Korisnika: {df['user'].nunique()}")
    print(f"Zlonamjernih user-dana: {df['malicious'].sum()} / {len(df)} "
          f"({100*df['malicious'].mean():.2f}%)")
