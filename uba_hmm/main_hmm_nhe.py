import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as C
from src import data_loading as D
from src import feature_engineering as FE
from src import nhe_scoring as NHE
from src.hmm_torch import BatchedGaussianHMM

N_STATES = 5
EM_ITERS = 30
N_RESTARTS = 3
KMEANS_INIT = True
HYBRID_GLOBAL = 0.0
N_BINS = 256
MIN_DAYS = 20
SEED = 42

HMM_FEATURES = [
    "roll7_std_num_usb_connects", "roll7_mean_num_usb_connects",
    "roll7_std_total_activity", "roll7_std_num_files_to_usb",
    "roll7_mean_num_files_to_usb", "num_usb_connects", "num_files_to_usb",
    "has_usb", "num_suspicious_http", "roll7_mean_num_suspicious_http",
    "num_after_hours_usb", "first_num_usb_connects", "z_num_usb_connects",
    "num_night_logons", "total_activity",
]

SCORE_MODE = "dominant"
OCC_THRESH = 0.10

def build_user_tensors(df, feat_cols):
    from sklearn.preprocessing import StandardScaler
    df = df.sort_values(["user", "day"]).copy()
    scaler = StandardScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols].astype(float))

    users = sorted(df["user"].unique())
    groups = {u: g for u, g in df.groupby("user")}
    lengths = {u: len(groups[u]) for u in users}
    users = [u for u in users if lengths[u] >= MIN_DAYS]
    Tmax = max(lengths[u] for u in users)
    U, F = len(users), len(feat_cols)

    obs = np.zeros((U, Tmax, F), dtype=np.float32)
    mask = np.zeros((U, Tmax), dtype=np.float32)
    days = []
    labels = np.zeros((U, Tmax), dtype=np.int64)
    for i, u in enumerate(users):
        g = groups[u]
        n = len(g)
        obs[i, :n] = g[feat_cols].values.astype(np.float32)
        mask[i, :n] = 1.0
        days.append(list(g["day"].values) + [None] * (Tmax - n))
        if "malicious" in g.columns:
            labels[i, :n] = g["malicious"].values
    return users, obs, mask, days, labels

def main():
    print("=" * 64)
    print(" NENADZIRANI HMM + NHE (hyper-binning) - CERT r4.2")
    print("=" * 64)
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[*] Uredjaj: GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("[*] Uredjaj: CPU")

    print("\n[1/5] Izgradnja dnevnih znacajki...")
    df = FE.build_daily_features(use_cache=True, verbose=True)
    feat_cols = [c for c in HMM_FEATURES if c in df.columns]
    print(f"      HMM znacajki: {len(feat_cols)} | {D.class_distribution(df)}")

    print("\n[2/5] Slaganje vremenskih nizova po korisniku...")
    users, obs, mask, days, labels = build_user_tensors(df, feat_cols)
    print(f"      Korisnika: {len(users)} | Tmax={obs.shape[1]} | znacajki={obs.shape[2]}")
    obs_t = torch.tensor(obs); mask_t = torch.tensor(mask)

    print(f"\n[3/5] Ucenje HMM-a ({N_STATES} stanja, {EM_ITERS} EM iteracija, "
          f"{N_RESTARTS} restarta, batchano)...")
    best_hmm, best_ll = None, -1e18
    for r in range(N_RESTARTS):
        hmm_r = BatchedGaussianHMM(n_users=len(users), n_states=N_STATES,
                                   n_features=len(feat_cols), device=device, seed=SEED + r)
        hist = hmm_r.fit(obs_t, mask_t, n_iter=EM_ITERS, kmeans=KMEANS_INIT, verbose=(r == 0))
        print(f"      restart {r+1}/{N_RESTARTS}: zavrsna log-izglednost={hist[-1]:.1f}")
        if hist[-1] > best_ll:
            best_ll, best_hmm = hist[-1], hmm_r
    hmm = best_hmm
    pi = hmm.stationary().cpu().numpy()
    ent = np.mean([-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(N_STATES) for p in pi])
    print(f"      Odabran najbolji HMM (log-izglednost={best_ll:.1f}, entropija stanja={ent:.3f})")
    if ent < 0.15:
        print("      NAPOMENA: niska entropija -> stanja kolabiraju; k-means init i vise znacajki pomazu.")

    print(f"\n[4/5] Racunanje log-vjerojatnosti (mod='{SCORE_MODE}') i NHE skorova...")
    if SCORE_MODE == "dominant":
        base_ll = hmm.dominant_state_loglik(obs_t, mask_t, occ_thresh=OCC_THRESH).cpu().numpy()
    elif SCORE_MODE == "marginal":
        base_ll = hmm.marginal_loglik(obs_t, mask_t).cpu().numpy()
    elif SCORE_MODE == "conditional":
        base_ll = hmm.conditional_loglik(obs_t, mask_t).cpu().numpy()
    else:
        cond = hmm.conditional_loglik(obs_t, mask_t).cpu().numpy()
        marg = hmm.marginal_loglik(obs_t, mask_t).cpu().numpy()
        base_ll = np.where(np.isnan(cond) | np.isnan(marg), np.nan, cond + marg)

    scores, nhe = NHE.compute_nhe_scores(base_ll, mask, n_bins=N_BINS)
    print(f"      Prosjecni NHE po korisniku: {np.nanmean(nhe):.3f}")

    if HYBRID_GLOBAL > 0:
        from scipy.stats import rankdata
        zc = [i for i, c in enumerate(feat_cols) if c.startswith("z_") or c.startswith("first_")]
        glob = np.full_like(scores, np.nan)
        for u in range(scores.shape[0]):
            m = mask[u] > 0
            if m.sum() > 0:
                glob[u, m] = np.abs(obs[u, m][:, zc]).mean(1)
        sh = scores.copy(); sg = glob.copy()
        valid = (mask > 0) & ~np.isnan(sh)
        rh = np.full(sh.shape, np.nan); rg = np.full(sg.shape, np.nan)
        rh[valid] = rankdata(sh[valid]); rg[valid] = rankdata(np.nan_to_num(sg[valid]))
        w = HYBRID_GLOBAL
        scores = w * rh + (1 - w) * rg
        print(f"      Hibrid: {w:.1f}*HMM + {1-w:.1f}*globalni outlier")

    U, T = scores.shape
    rows_s, rows_y = [], []
    for u in range(U):
        for t in range(T):
            if mask[u, t] > 0 and not np.isnan(scores[u, t]):
                rows_s.append(scores[u, t]); rows_y.append(labels[u, t])
    s = np.array(rows_s); y = np.array(rows_y)

    print("\n[5/5] Evaluacija (oznake koristene samo za mjerenje)...")
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(y, s) if y.sum() > 0 else float("nan")
    pr = average_precision_score(y, s) if y.sum() > 0 else float("nan")
    print(f"      Ukupno (korisnik,dan): {len(y)} | zlonamjernih: {int(y.sum())}")
    print(f"      ROC-AUC: {auc:.3f}   PR-AUC: {pr:.3f}")

    order = np.argsort(-s)
    total_mal = int(y.sum())
    print("\n      Top N | pogodaka | precision@N | recall@N")
    print("      " + "-" * 46)
    for n in [50, 100, 200, 500, 1000, 2000]:
        if n > len(s):
            break
        hits = int(y[order[:n]].sum())
        print(f"      {n:5d} | {hits:8d} | {hits/n:11.3f} | {hits/max(total_mal,1):8.3f}")

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    json.dump({"roc_auc": auc, "pr_auc": pr, "n_states": N_STATES,
               "em_iters": EM_ITERS, "n_bins": N_BINS,
               "mean_nhe": float(np.nanmean(nhe))},
              open(os.path.join(C.RESULTS_DIR, "metrics_hmm_nhe.json"), "w"), indent=2)
    print("\n" + "=" * 64)
    print(" GOTOVO. Rezultat u results/metrics_hmm_nhe.json")
    print("=" * 64)

if __name__ == "__main__":
    main()
