import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, accuracy_score, confusion_matrix,
                             precision_recall_curve)

from .transformer_torch import TransformerClassifier, FocalLoss

def get_device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        print(f"[*] Uredjaj: GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("[*] Uredjaj: CPU (GPU nije dostupan)")
    return dev

def _loader(X, y, batch_size, shuffle, device):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=(device.type == "cuda"))

def train_model(X, y, tr_idx, va_idx, n_features, seq_len, params,
                device=None, epochs=100, batch_size=128, patience=15,
                use_focal=False, use_amp=True, verbose=True):
    if device is None:
        device = get_device()

    model = TransformerClassifier(
        n_features=n_features, seq_len=seq_len,
        d_model=params.get("d_model", 64), num_heads=params.get("num_heads", 4),
        ff_dim=params.get("ff_dim", 128), num_layers=params.get("num_layers", 2),
        dropout=params.get("dropout", 0.1),
    ).to(device)

    pos = float(y[tr_idx].sum()); neg = float(len(tr_idx) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    if use_focal:
        criterion = FocalLoss(gamma=2.0, alpha=0.25)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params.get("learning_rate", 1e-3), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    train_loader = _loader(X[tr_idx], y[tr_idx], batch_size, True, device)
    Xva = torch.tensor(X[va_idx], dtype=torch.float32, device=device)
    yva = y[va_idx]

    best_pr, best_state, wait = -1.0, None, 0
    history = {"loss": [], "val_pr_auc": [], "val_roc_auc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                p = torch.sigmoid(model(Xva)).float().cpu().numpy()
        pr = average_precision_score(yva, p) if yva.sum() > 0 else 0.0
        roc = roc_auc_score(yva, p) if len(set(yva)) > 1 else float("nan")
        scheduler.step(pr)
        history["loss"].append(total / len(tr_idx))
        history["val_pr_auc"].append(pr)
        history["val_roc_auc"].append(roc)

        if pr > best_pr:
            best_pr, best_state, wait = pr, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"      epoha {epoch:3d}/{epochs}  gubitak={history['loss'][-1]:.4f}  "
                  f"val_PR-AUC={pr:.3f}  val_ROC-AUC={roc:.3f}")

        if wait >= patience:
            if verbose:
                print(f"      rani prekid na epohi {epoch} (najbolji val PR-AUC={best_pr:.3f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_pr

@torch.no_grad()
def predict_proba(model, X, device, batch_size=512, use_amp=True):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            out.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return np.concatenate(out)

def evaluate(model, X_te, y_te, device):
    probs = predict_proba(model, X_te, device)
    prec_c, rec_c, thr_c = precision_recall_curve(y_te, probs)
    f1s = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-9)
    best_thr = thr_c[np.argmax(f1s[:-1])] if len(thr_c) else 0.5
    preds = (probs >= best_thr).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_te, preds),
        "precision": precision_score(y_te, preds, zero_division=0),
        "recall": recall_score(y_te, preds, zero_division=0),
        "f1": f1_score(y_te, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_te, probs) if y_te.sum() > 0 else float("nan"),
        "pr_auc": average_precision_score(y_te, probs) if y_te.sum() > 0 else float("nan"),
        "threshold": float(best_thr),
    }
    cm = confusion_matrix(y_te, preds)
    return metrics, cm, probs

def permutation_importance(model, X_te, y_te, feat_cols, device, n_repeats=2):
    base = average_precision_score(y_te, predict_proba(model, X_te, device))
    rng = np.random.default_rng(42)
    imp = {}
    for f, name in enumerate(feat_cols):
        drops = []
        for _ in range(n_repeats):
            Xp = X_te.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, :, f] = Xp[perm, :, f]
            drops.append(base - average_precision_score(y_te, predict_proba(model, Xp, device)))
        imp[name] = float(np.mean(drops))
    return dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))
