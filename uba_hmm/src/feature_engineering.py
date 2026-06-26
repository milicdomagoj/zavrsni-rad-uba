import os
import numpy as np
import pandas as pd

from . import config as C
from . import data_loading as D

def _logon_daily():
    df = pd.read_csv(D.find_file("logon"),
                     usecols=D.pick_cols("logon", ["date", "user", "pc", "activity"]))
    dt = pd.to_datetime(df["date"], format=C.DATE_FMT, errors="coerce")
    df = df[dt.notna()].copy(); dt = dt[dt.notna()]
    df["day"] = dt.dt.date
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    logon = df[df["activity"] == "Logon"].copy()
    logon["after"] = ((logon["hour"] < C.WORK_START) | (logon["hour"] >= C.WORK_END)).astype(int)
    logon["night"] = ((logon["hour"] >= 22) | (logon["hour"] < 6)).astype(int)
    logon["weekend"] = (logon["weekday"] >= 5).astype(int)
    g = logon.groupby(["user", "day"]).agg(
        num_logons=("activity", "size"),
        num_after_hours_logons=("after", "sum"),
        num_night_logons=("night", "sum"),
        num_weekend_logons=("weekend", "sum"),
        num_distinct_pcs=("pc", "nunique"),
    ).reset_index()
    return g

def _device_daily():
    df = pd.read_csv(D.find_file("device"),
                     usecols=D.pick_cols("device", ["date", "user", "activity"]))
    dt = pd.to_datetime(df["date"], format=C.DATE_FMT, errors="coerce")
    df = df[dt.notna()].copy(); dt = dt[dt.notna()]
    df["day"] = dt.dt.date
    df["after"] = ((dt.dt.hour < C.WORK_START) | (dt.dt.hour >= C.WORK_END)).astype(int)
    conn = df[df["activity"] == "Connect"].copy()
    g = conn.groupby(["user", "day"]).agg(
        num_usb_connects=("activity", "size"),
        num_after_hours_usb=("after", "sum"),
    ).reset_index()
    return g

def _file_daily():
    df = pd.read_csv(D.find_file("file"),
                     usecols=D.pick_cols("file", ["date", "user", "filename"]))
    df["day"] = pd.to_datetime(df["date"], format=C.DATE_FMT, errors="coerce").dt.date
    df = df.dropna(subset=["day"])
    g = df.groupby(["user", "day"]).agg(num_file_events=("filename", "size")).reset_index()
    g["num_files_to_usb"] = g["num_file_events"]
    return g

def _email_daily():
    cols = D.pick_cols("email", ["date", "user", "to", "cc", "bcc", "from", "size", "attachments"])
    parts = []
    for ch in pd.read_csv(D.find_file("email"), usecols=cols, chunksize=C.CHUNK_SIZE):
        ch["day"] = ch["date"].astype(str).str.slice(0, 10)
        rec = ch["to"].fillna("").astype(str)
        for c in ("cc", "bcc"):
            if c in ch.columns:
                rec = rec + ";" + ch[c].fillna("").astype(str)
        ch["external"] = (~rec.str.contains(C.INTERNAL_DOMAIN, na=False)).astype(int)
        ch["n_recipients"] = rec.str.count("@").clip(lower=0)
        ch["size"] = pd.to_numeric(ch.get("size", 0), errors="coerce").fillna(0)
        att = ch.get("attachments", 0)
        ch["attachments"] = pd.to_numeric(att, errors="coerce").fillna(0)
        parts.append(ch.groupby(["user", "day"]).agg(
            num_emails=("external", "size"),
            num_external_emails=("external", "sum"),
            total_email_size=("size", "sum"),
            num_attachments=("attachments", "sum"),
            total_recipients=("n_recipients", "sum"),
        ).reset_index())
    out = pd.concat(parts, ignore_index=True).groupby(["user", "day"], as_index=False).sum()
    return out

def _http_daily():
    cols = D.pick_cols("http", ["date", "user", "url"])
    pat = "|".join(C.SUSPICIOUS_SITES)
    parts = []
    for ch in pd.read_csv(D.find_file("http"), usecols=cols, chunksize=C.CHUNK_SIZE):
        ch["day"] = ch["date"].astype(str).str.slice(0, 10)
        ch["susp"] = ch["url"].astype(str).str.contains(pat, case=False, na=False).astype(int)
        parts.append(ch.groupby(["user", "day"]).agg(
            num_http=("susp", "size"),
            num_suspicious_http=("susp", "sum"),
        ).reset_index())
    out = pd.concat(parts, ignore_index=True).groupby(["user", "day"], as_index=False).sum()
    return out

BASE_NUMERIC = [
    "num_logons", "num_after_hours_logons", "num_night_logons", "num_weekend_logons",
    "num_distinct_pcs", "num_usb_connects", "num_after_hours_usb",
    "num_file_events", "num_files_to_usb",
    "num_emails", "num_external_emails", "total_email_size", "num_attachments",
    "total_recipients", "num_http", "num_suspicious_http",
]

def build_daily_features(use_cache=True, verbose=True):
    csv_cache = C.DAILY_CACHE.replace(".parquet", ".csv")
    if use_cache:
        try:
            df = pd.read_parquet(C.DAILY_CACHE)
            if verbose:
                print(f"[cache] ucitano {len(df)} redaka iz {C.DAILY_CACHE}")
            return df
        except Exception:
            pass
        if os.path.exists(csv_cache):
            df = pd.read_csv(csv_cache)
            df["day"] = pd.to_datetime(df["day"])
            if verbose:
                print(f"[cache] ucitano {len(df)} redaka iz {csv_cache}")
            return df

    if verbose:
        print("[*] Racunanje dnevnih znacajki iz sirovih zapisa...")
    logon = _logon_daily()
    device = _device_daily()
    file_df = _file_daily()
    email = _email_daily()
    http = _http_daily()

    for d in (email, http):
        d["day"] = pd.to_datetime(d["day"], format=C.DAY_FMT, errors="coerce").dt.date
        d.dropna(subset=["day"], inplace=True)

    df = logon
    for o in (device, file_df, email, http):
        df = df.merge(o, on=["user", "day"], how="outer")
    for c in BASE_NUMERIC:
        if c not in df.columns:
            df[c] = 0
    df[BASE_NUMERIC] = df[BASE_NUMERIC].fillna(0)
    df["day"] = pd.to_datetime(df["day"])
    df = df.sort_values(["user", "day"]).reset_index(drop=True)

    df["total_activity"] = df[["num_logons", "num_usb_connects", "num_file_events",
                               "num_emails", "num_http"]].sum(axis=1)

    df["is_weekend"] = (df["day"].dt.weekday >= 5).astype(int)
    df["has_night_activity"] = (df["num_night_logons"] > 0).astype(int)
    df["has_usb"] = (df["num_usb_connects"] > 0).astype(int)
    df["has_suspicious_http"] = (df["num_suspicious_http"] > 0).astype(int)
    df["has_external_email"] = (df["num_external_emails"] > 0).astype(int)

    if verbose:
        print("[*] Dodavanje naprednih ponasajnih znacajki (z-score, rolling, novelty)...")

    for col in BASE_NUMERIC + ["total_activity"]:
        g = df.groupby("user")[col]
        m = g.transform("mean")
        s = g.transform("std").replace(0, np.nan)
        df["z_" + col] = ((df[col] - m) / s).fillna(0.0)

    for col in ["total_activity", "num_usb_connects", "num_files_to_usb",
                "num_external_emails", "num_suspicious_http"]:
        grp = df.groupby("user")[col]
        df["roll7_mean_" + col] = grp.transform(lambda s: s.rolling(7, min_periods=1).mean())
        df["roll7_std_" + col] = grp.transform(
            lambda s: s.rolling(7, min_periods=1).std()).fillna(0.0)

    df["days_since_prev"] = df.groupby("user")["day"].diff().dt.days.fillna(0)

    df["active_today"] = (df["total_activity"] > 0).astype(int)
    df["activity_frequency"] = (df.groupby("user")["active_today"].cumsum() /
                                (df.groupby("user").cumcount() + 1))

    for col in ["num_usb_connects", "num_night_logons", "num_suspicious_http"]:
        nz = (df[col] > 0).astype(int)
        prior = df.groupby("user")[col].apply(
            lambda s: (s > 0).astype(int).cumsum().shift(fill_value=0)
        ).reset_index(level=0, drop=True)
        df["first_" + col] = ((nz == 1) & (prior == 0)).astype(int)

    ins = D.load_insiders()
    mal = set()
    for _, w in ins.iterrows():
        for d in pd.date_range(w["start"].normalize(), w["end"].normalize(), freq="D"):
            mal.add((w["user"], d.date()))
    df["malicious"] = [1 if (u, d.date()) in mal else 0
                       for u, d in zip(df["user"], df["day"])]

    if use_cache:
        try:
            df.to_parquet(C.DAILY_CACHE, index=False)
            if verbose:
                print(f"[cache] spremljeno u {C.DAILY_CACHE}")
        except Exception:
            try:
                df.to_csv(csv_cache, index=False)
                if verbose:
                    print(f"[cache] spremljeno u {csv_cache} (CSV; za parquet instaliraj pyarrow)")
            except Exception as e:
                print("  (upozorenje: cache nije spremljen:", e, ")")

    if verbose:
        print(f"[*] Gotovo. Matrica: {df.shape[0]} redaka x {df.shape[1]} stupaca")
    return df

def feature_columns(df):
    exclude = {"user", "day", "malicious", "active_today"}
    return [c for c in df.columns if c not in exclude and
            pd.api.types.is_numeric_dtype(df[c])]

if __name__ == "__main__":
    df = build_daily_features(use_cache=False, verbose=True)
    cols = feature_columns(df)
    print(f"Broj znacajki: {len(cols)}")
    print(cols)
