import numpy as np

def _robust_norm(values):
    med = np.nanmedian(values)
    return med - values

def compute_nhe_scores(base_ll, valid_mask, n_bins=256, alpha=1.0):
    U, T = base_ll.shape
    scores = np.full((U, T), np.nan, dtype=np.float64)
    nhe = np.full(U, np.nan, dtype=np.float64)

    valid = valid_mask > 0
    ll_valid = base_ll[valid]
    if ll_valid.size == 0:
        return scores, nhe
    gmax = np.nanmax(ll_valid)

    lo, hi = np.nanpercentile(ll_valid, 0.5), np.nanpercentile(ll_valid, 99.5)
    edges = np.linspace(lo, hi, n_bins + 1)
    log_B = np.log(n_bins)

    for u in range(U):
        m = valid[u]
        if m.sum() < 5:
            continue
        llu = base_ll[u].copy()
        hist, _ = np.histogram(np.clip(llu[m], lo, hi), bins=edges)
        p = hist / max(hist.sum(), 1)
        nz = p[p > 0]
        nhe_u = float(-(nz * np.log(nz)).sum() / log_B) if len(nz) else 0.0
        nhe[u] = nhe_u
        base = gmax - llu
        s = base * (1.0 + alpha * (1.0 - nhe_u))
        s[~m] = np.nan
        scores[u] = s
    return scores, nhe

def flatten_scores(scores, valid_mask, users, days):
    out_u, out_d, out_s = [], [], []
    U, T = scores.shape
    for u in range(U):
        for t in range(T):
            if valid_mask[u, t] > 0 and not np.isnan(scores[u, t]):
                out_u.append(users[u]); out_d.append(days[u][t]); out_s.append(scores[u, t])
    return np.array(out_u), np.array(out_d), np.array(out_s, dtype=float)
