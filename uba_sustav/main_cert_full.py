import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.cert_full import build_full_feature_matrix
from src.cert_features_baseline import add_relative_features
from src.model import run_all_models, run_isolation_forest, prepare_matrix
from src import visualize

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

def main(contamination=0.01):
    print("=" * 60)
    print(" UBA SUSTAV - CIJELI CERT r4.2 dataset (svi korisnici)")
    print("=" * 60)

    print("\n[1/4] Izgradnja matrice znacajki...")
    feats, abs_cols = build_full_feature_matrix()
    print(f"      Matrica: {feats.shape[0]} (user-dan), {feats['user'].nunique()} korisnika")
    print(f"      Zlonamjernih: {feats['malicious'].sum()} / {len(feats)} "
          f"({100*feats['malicious'].mean():.2f}%)")

    if feats["malicious"].sum() == 0:
        print("\n  UPOZORENJE: nema zlonamjernih oznaka! Provjeri insiders.csv.")
        return

    print("\n[2/4] Racunanje relativnih znacajki po korisniku (baseline)...")
    feats, rel_cols = add_relative_features(feats)
    print(f"      Relativne znacajke ({len(rel_cols)}): {rel_cols}")

    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    print("\n[3/4] Evaluacija - APSOLUTNE znacajke (stari pristup):")
    metrics_abs, _, _ = run_all_models(feats, abs_cols, contamination=contamination)
    print(metrics_abs.to_string())

    print("\n[3/4] Evaluacija - RELATIVNE znacajke (po korisniku):")
    metrics_rel, score_dict, y = run_all_models(feats, rel_cols, contamination=contamination)
    print(metrics_rel.to_string())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_abs.to_csv(os.path.join(RESULTS_DIR, "metrike_apsolutne.csv"))
    metrics_rel.to_csv(os.path.join(RESULTS_DIR, "metrike_relativne.csv"))

    print("\n[4/4] Generiranje grafova (relativne znacajke)...")
    visualize.plot_roc_curves(y, score_dict, filename="roc_krivulje_cert_full.png")
    visualize.plot_metrics_bar(metrics_rel, filename="usporedba_metrika_cert_full.png")
    visualize.plot_score_distribution(y, score_dict["Isolation Forest"],
                                      filename="distribucija_scoreova_cert_full.png")
    X, _, _ = prepare_matrix(feats, rel_cols)
    _, if_preds = run_isolation_forest(X, contamination)
    visualize.plot_confusion(y, if_preds, filename="matrica_konfuzije_cert_full.png")

    top = visualize.top_anomalies(feats, score_dict["Isolation Forest"], n=20)
    top.to_csv(os.path.join(RESULTS_DIR, "top_anomalije_cert_full.csv"), index=False)
    print("\n      Top 10 detektiranih anomalija (relativne znacajke):")
    print(top.head(10).to_string(index=False))
    hits = int(top.head(20)["malicious"].sum())
    print(f"\n      Od 20 najanomalnijih, stvarnih insajdera: {hits}")

    print("\n" + "=" * 60)
    print(" Gotovo. Usporedi metrike_apsolutne.csv i metrike_relativne.csv")
    print("=" * 60)

if __name__ == "__main__":
    main()
