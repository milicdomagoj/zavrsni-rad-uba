

import numpy as np
import pandas as pd

# Znacajke za koje racunamo z-score po korisniku
ZSCORE_BASE = ["num_after_hours_logons", "num_usb_connects", "num_after_hours_usb",
               "num_files_to_usb", "num_external_emails", "total_email_size",
               "num_suspicious_http", "num_http"]

# Znacajke za "prvi put" indikatore (mapiraju na CERT scenarije)
NOVELTY_BASE = ["num_usb_connects", "num_after_hours_logons", "num_suspicious_http"]

# Apsolutne znacajke koje zadrzavamo (same po sebi indikativne)
KEEP_ABSOLUTE = ["num_suspicious_http"]


def add_relative_features(df):
   
    df = df.sort_values(["user", "day"]).reset_index(drop=True)
    model_cols = []

    for col in ZSCORE_BASE:
        if col not in df.columns:
            continue
        grp = df.groupby("user")[col]
        mean = grp.transform(lambda x: x.expanding().mean().shift())
        std = grp.transform(lambda x: x.expanding().std().shift()).replace(0, np.nan)
        z = (df[col] - mean) / std
        zcol = "z_" + col
        df[zcol] = z.fillna(0.0)
        model_cols.append(zcol)

    for col in NOVELTY_BASE:
        if col not in df.columns:
            continue
        nonzero = (df[col] > 0).astype(int)
        # broj prethodnih dana (prije danasnjeg) s nenultom vrijednoscu
        prior = df.groupby("user")[col].apply(
            lambda s: (s > 0).astype(int).cumsum().shift(fill_value=0)
        ).reset_index(level=0, drop=True)
        first_col = "first_" + col
        df[first_col] = ((nonzero == 1) & (prior == 0)).astype(int)
        model_cols.append(first_col)

    for col in KEEP_ABSOLUTE:
        if col in df.columns and col not in model_cols:
            model_cols.append(col)

    return df, model_cols


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from cert_full import build_full_feature_matrix
    df, _ = build_full_feature_matrix()
    df, cols = add_relative_features(df)
    print(f"Relativne znacajke ({len(cols)}): {cols}")
    print(df[["user", "day", "malicious"] + cols].head())
