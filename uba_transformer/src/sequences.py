import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as C

def make_sequences(df, feat_cols, seq_len, stride=C.STRIDE, label_mode="any"):
    Xs, ys, groups = [], [], []
    F = len(feat_cols)
    for user, g in df.groupby("user"):
        g = g.sort_values("day").reset_index(drop=True)
        if len(g) < seq_len:
            continue
        vals = g[feat_cols].values.astype(np.float32)
        mal = g["malicious"].values
        for s in range(0, len(g) - seq_len + 1, stride):
            window = mal[s:s + seq_len]
            if label_mode == "last":
                label = int(window[-1])
            else:
                label = int(window.max())
            Xs.append(vals[s:s + seq_len])
            ys.append(label)
            groups.append(user)
    X = np.stack(Xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    groups = np.array(groups)
    return X, y, groups

def split_by_user(groups, y, test_frac=C.TEST_FRAC, val_frac=C.VAL_FRAC, seed=C.SEED):
    rng = np.random.default_rng(seed)
    users = np.unique(groups)
    insider_users = np.unique(groups[y == 1])
    normal_users = np.array([u for u in users if u not in set(insider_users)])
    rng.shuffle(insider_users); rng.shuffle(normal_users)

    def carve(arr, frac):
        k = int(round(len(arr) * frac))
        return set(arr[:k])

    test_u = carve(insider_users, test_frac) | carve(normal_users, test_frac)
    rem_ins = [u for u in insider_users if u not in test_u]
    rem_norm = [u for u in normal_users if u not in test_u]
    val_u = (set(rem_ins[:int(round(len(rem_ins) * val_frac))]) |
             set(rem_norm[:int(round(len(rem_norm) * val_frac))]))

    te = np.array([g in test_u for g in groups])
    va = np.array([g in val_u for g in groups])
    tr = ~(te | va)
    return tr, va, te

def kfold_by_user(groups, y, n_folds=5, val_frac=0.15, seed=C.SEED):
    rng = np.random.default_rng(seed)
    users = np.unique(groups)
    insider_users = np.unique(groups[y == 1])
    normal_users = np.array([u for u in users if u not in set(insider_users)])
    rng.shuffle(insider_users); rng.shuffle(normal_users)

    ins_folds = np.array_split(insider_users, n_folds)
    norm_folds = np.array_split(normal_users, n_folds)

    for k in range(n_folds):
        test_u = set(ins_folds[k]) | set(norm_folds[k])
        rem_ins = [u for f in range(n_folds) if f != k for u in ins_folds[f]]
        rem_norm = [u for f in range(n_folds) if f != k for u in norm_folds[f]]
        rng.shuffle(rem_ins); rng.shuffle(rem_norm)
        val_u = (set(rem_ins[:int(round(len(rem_ins) * val_frac))]) |
                 set(rem_norm[:int(round(len(rem_norm) * val_frac))]))
        te = np.array([g in test_u for g in groups])
        va = np.array([g in val_u for g in groups])
        tr = ~(te | va)
        yield k, tr, va, te

def scale_sequences(X, tr_mask, feat_dim):
    scaler = StandardScaler()
    flat_tr = X[tr_mask].reshape(-1, feat_dim)
    scaler.fit(flat_tr)
    Xs = scaler.transform(X.reshape(-1, feat_dim)).reshape(X.shape).astype(np.float32)
    return Xs, scaler

def compute_class_weight(y_tr):
    pos = float(y_tr.sum()); neg = float(len(y_tr) - pos)
    total = pos + neg
    return {0: total / (2 * neg), 1: total / (2 * pos)}

def subsample_negatives(tr_mask, y, neg_per_pos=8, seed=C.SEED):
    rng = np.random.default_rng(seed)
    tr_idx = np.where(tr_mask)[0]
    pos = tr_idx[y[tr_idx] == 1]
    neg = tr_idx[y[tr_idx] == 0]
    keep = rng.choice(neg, size=min(len(neg), neg_per_pos * len(pos)), replace=False)
    sel = np.concatenate([pos, keep])
    rng.shuffle(sel)
    return sel
