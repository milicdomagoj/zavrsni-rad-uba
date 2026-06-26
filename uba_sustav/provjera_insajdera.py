import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from src.cert_full import build_full_feature_matrix
from src.cert_features_baseline import add_relative_features
from src.model import prepare_matrix, run_isolation_forest

print("[*] Ucitavanje matrice znacajki (iz cachea)...")
feats, _ = build_full_feature_matrix()
feats, rel_cols = add_relative_features(feats)

print("[*] Treniranje Isolation Forest na relativnim znacajkama...")
X, y, _ = prepare_matrix(feats, rel_cols)
scores, _ = run_isolation_forest(X, contamination=0.01)

feats = feats.copy()
feats["anomaly_score"] = scores
ranked = feats.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

total_malicious = int(feats["malicious"].sum())
total_insiders = feats[feats["malicious"] == 1]["user"].nunique()

print("\n" + "=" * 64)
print(f" Ukupno zlonamjernih dana: {total_malicious}")
print(f" Ukupno stvarnih insajdera (korisnika): {total_insiders}")
print("=" * 64)
print(f"\n{'Top N':>7} | {'pogodaka':>8} | {'precision@N':>11} | "
      f"{'recall@N':>9} | {'razlicitih insajdera':>20}")
print("-" * 70)
for n in [50, 100, 200, 500, 1000, 2000]:
    top = ranked.head(n)
    hits = int(top["malicious"].sum())
    prec = hits / n
    rec = hits / total_malicious
    uniq = top[top["malicious"] == 1]["user"].nunique()
    print(f"{n:>7} | {hits:>8} | {prec:>11.3f} | {rec:>9.3f} | {uniq:>20}")

print("\nTumacenje:")
print("  precision@N = koliko je od top N prijava stvarno zlonamjerno")
print("  recall@N    = koliki udio SVIH zlonamjernih dana uhvatimo u top N")
print("  Ako analiticar pregledava listu od vrha, ovo pokazuje koliko brzo")
print("  nailazi na prave insajdere.")
