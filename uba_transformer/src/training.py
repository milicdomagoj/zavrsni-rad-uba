import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, accuracy_score, confusion_matrix,
                             roc_curve, precision_recall_curve)

from . import config as C
from . import sequences as S
from .transformer_model import build_transformer, build_transformer_focal

def enable_mixed_precision():
    if C.USE_MIXED_PRECISION:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("[*] Mixed precision ukljucen (mixed_float16).")
        except Exception as e:
            print("  (mixed precision nije moguc:", e, ")")

def make_callbacks(model_path):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                         patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=6, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_pr_auc", mode="max",
                                           save_best_only=True),
    ]

def select_best_seq_length(df, feat_cols, lengths=C.SEQ_LENGTHS, quick_epochs=15):
    results = {}
    for L in lengths:
        X, y, groups = S.make_sequences(df, feat_cols, seq_len=L)
        if y.sum() < 5:
            print(f"  duljina {L}: premalo pozitivnih, preskacem")
            continue
        tr, va, te = S.split_by_user(groups, y)
        X, _ = S.scale_sequences(X, tr, len(feat_cols))
        sel = S.subsample_negatives(tr, y)
        model = build_transformer(L, len(feat_cols), d_model=48, num_heads=3,
                                  ff_dim=96, num_layers=2, dropout=0.1)
        cw = S.compute_class_weight(y[sel])
        model.fit(X[sel], y[sel], validation_data=(X[va], y[va]),
                  epochs=quick_epochs, batch_size=C.BATCH_SIZE, class_weight=cw, verbose=0)
        p = model.predict(X[va], verbose=0).ravel()
        pr = average_precision_score(y[va], p) if y[va].sum() > 0 else 0.0
        results[L] = pr
        print(f"  duljina {L:3d}: PR-AUC(val) = {pr:.3f}")
    best = max(results, key=results.get)
    print(f"[*] Najbolja duljina sekvence: {best}")
    return best, results

def optimize_hyperparams(X, y, tr, va, feat_dim, seq_len, n_trials=C.OPTUNA_TRIALS):
    import optuna

    sel = S.subsample_negatives(tr, y)
    cw = S.compute_class_weight(y[sel])

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [32, 48, 64, 96])
        num_heads = trial.suggest_categorical("num_heads", [2, 3, 4])
        ff_dim = trial.suggest_categorical("ff_dim", [64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.05, 0.4)
        lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        if d_model % num_heads != 0:
            return 0.0
        model = build_transformer(seq_len, feat_dim, d_model, num_heads, ff_dim,
                                  num_layers, dropout, lr)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                              patience=8, restore_best_weights=True)
        model.fit(X[sel], y[sel], validation_data=(X[va], y[va]),
                  epochs=C.OPTUNA_EPOCHS, batch_size=C.BATCH_SIZE,
                  class_weight=cw, callbacks=[es], verbose=0)
        p = model.predict(X[va], verbose=0).ravel()
        return average_precision_score(y[va], p) if y[va].sum() > 0 else 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[*] Najbolji PR-AUC (val): {study.best_value:.3f}")
    print(f"[*] Najbolji hiperparametri: {study.best_params}")
    return study.best_params, study

def evaluate(model, X_te, y_te, results_dir=C.RESULTS_DIR, prefix="transformer"):
    probs = model.predict(X_te, verbose=0).ravel()
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
    np.save(os.path.join(results_dir, f"{prefix}_probs.npy"), probs)
    np.save(os.path.join(results_dir, f"{prefix}_labels.npy"), y_te)
    return metrics, cm, probs

def permutation_importance(model, X_te, y_te, feat_cols, n_repeats=2):
    base = average_precision_score(y_te, model.predict(X_te, verbose=0).ravel())
    rng = np.random.default_rng(C.SEED)
    importances = {}
    for f, name in enumerate(feat_cols):
        drops = []
        for _ in range(n_repeats):
            Xp = X_te.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, :, f] = Xp[perm, :, f]
            p = model.predict(Xp, verbose=0).ravel()
            drops.append(base - average_precision_score(y_te, p))
        importances[name] = float(np.mean(drops))
    return dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

def attention_summary(model, X_sample):
    import tensorflow as tf
    block = None; block_idx = None
    for i, layer in enumerate(model.layers):
        if layer.__class__.__name__ == "TransformerEncoderBlock":
            block = layer; block_idx = i; break
    if block is None or block_idx is None:
        return None
    feeder = tf.keras.Model(model.inputs, model.layers[block_idx - 1].output)
    x = feeder(X_sample, training=False)
    _, scores = block.att(x, x, return_attention_scores=True, training=False)
    avg = tf.reduce_mean(scores, axis=[0, 1]).numpy()
    return avg.mean(axis=0)
