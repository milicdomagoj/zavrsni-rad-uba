import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as C
from src import data_loading as D
from src import feature_engineering as FE
from src import sequences as S
from src import visualization as V
from src import training_torch as TT

SEQ_LEN = 30
STRIDE = 7
LABEL_MODE = "last"
EPOCHS = 120
BATCH_SIZE = 256
USE_ALL_TRAIN = True
NEG_PER_POS = 30
USE_FOCAL = False
N_FOLDS = 5

PARAMS = {
    "d_model": 96,
    "num_heads": 4,
    "ff_dim": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 1e-3,
}

def main():
    print("=" * 64)
    print(" TRANSFORMER ENCODER (PyTorch / GPU) - CERT r4.2")
    print(" Stroza, realnija evaluacija (kraci prozor + oznaka zadnjeg dana + k-fold)")
    print("=" * 64)
    np.random.seed(C.SEED); torch.manual_seed(C.SEED)
    device = TT.get_device()

    print("\n[1/6] Eksploratorna analiza (EDA)...")
    D.explore_dataset(verbose=True)

    print("\n[2/6] Izgradnja dnevnih znacajki...")
    df = FE.build_daily_features(use_cache=True, verbose=True)
    feat_cols = FE.feature_columns(df)
    print(f"      Znacajki: {len(feat_cols)} | {D.class_distribution(df)}")
    V.plot_class_distribution(df)

    print(f"\n[3/6] Izgradnja sljedova (L={SEQ_LEN}, korak={STRIDE}, oznaka='{LABEL_MODE}')...")
    X, y, groups = S.make_sequences(df, feat_cols, seq_len=SEQ_LEN,
                                    stride=STRIDE, label_mode=LABEL_MODE)
    print(f"      X={X.shape} | pozitivnih prozora={int(y.sum())}/{len(y)} "
          f"({100*y.mean():.3f}%)")
    if y.sum() < 20:
        print("      UPOZORENJE: malo pozitivnih prozora; razmotri label_mode='any' ili veci STRIDE.")

    use_kfold = N_FOLDS and N_FOLDS >= 2
    folds = (S.kfold_by_user(groups, y, n_folds=N_FOLDS) if use_kfold
             else [(0,) + S.split_by_user(groups, y)])
    if use_kfold:
        print(f"\n[4/6] {N_FOLDS}-struka unakrsna provjera (rezultat = prosjek svih foldova)...")
    else:
        print("\n[4/6] Jedna podjela trening/val/test po korisniku...")

    fold_metrics = []
    last_probs = last_y = last_cm = None
    last_model = None; last_feat_imp = None

    for k, tr, va, te in folds:
        Xs, _ = S.scale_sequences(X, tr, len(feat_cols))
        tr_idx = np.where(tr)[0] if USE_ALL_TRAIN else S.subsample_negatives(tr, y, neg_per_pos=NEG_PER_POS)
        va_idx = np.where(va)[0]
        te_idx = np.where(te)[0]
        print(f"\n   --- Fold {k+1}/{len(folds) if not use_kfold else N_FOLDS} ---")
        print(f"      trening={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)} | "
              f"poz(test)={int(y[te_idx].sum())}")
        if y[te_idx].sum() == 0 or y[va_idx].sum() == 0:
            print("      (preskacem fold bez pozitivnih u val/test)")
            continue

        model, history, best_pr = TT.train_model(
            Xs, y, tr_idx, va_idx, n_features=len(feat_cols), seq_len=SEQ_LEN,
            params=PARAMS, device=device, epochs=EPOCHS, batch_size=BATCH_SIZE,
            use_focal=USE_FOCAL, use_amp=True, verbose=False,
        )
        metrics, cm, probs = TT.evaluate(model, Xs[te_idx], y[te_idx], device)
        fold_metrics.append(metrics)
        print(f"      Fold rezultat: ROC-AUC={metrics['roc_auc']:.3f}  "
              f"PR-AUC={metrics['pr_auc']:.3f}  F1={metrics['f1']:.3f}  "
              f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}")
        last_probs, last_y, last_cm, last_model = probs, y[te_idx], cm, model
        last_Xs_te, last_te_idx = Xs, te_idx

    print("\n[5/6] Sazetak rezultata (prosjek +/- st.dev. po foldovima)...")
    if not fold_metrics:
        print("      Nema valjanih foldova."); return
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    summary = {}
    for kk in keys:
        vals = np.array([m[kk] for m in fold_metrics], dtype=float)
        summary[kk] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
        print(f"        {kk:12s}: {summary[kk]['mean']:.4f}  +/- {summary[kk]['std']:.4f}")
    with open(os.path.join(C.RESULTS_DIR, "metrics_torch_cv.json"), "w") as f:
        json.dump({"folds": fold_metrics, "summary": summary,
                   "config": {"seq_len": SEQ_LEN, "stride": STRIDE,
                              "label_mode": LABEL_MODE, "n_folds": N_FOLDS}}, f, indent=2)

    V.plot_roc(last_y, last_probs, fname="roc_curve_torch.png")
    V.plot_pr(last_y, last_probs, fname="pr_curve_torch.png")
    V.plot_confusion(last_cm, fname="confusion_matrix_torch.png")

    order = np.argsort(-last_probs)
    print("\n      Top N (zadnji fold) | pogodaka | precision@N | recall@N")
    tot = int(last_y.sum())
    for nn_ in [10, 25, 50, 100]:
        if nn_ > len(last_probs):
            break
        hits = int(last_y[order[:nn_]].sum())
        print(f"      {nn_:5d} | {hits:8d} | {hits/nn_:11.3f} | {hits/max(tot,1):8.3f}")

    print("\n[6/6] Vaznost znacajki + spremanje modela (zadnji fold)...")
    imp = TT.permutation_importance(last_model, last_Xs_te[last_te_idx], last_y,
                                    feat_cols, device, n_repeats=2)
    with open(os.path.join(C.RESULTS_DIR, "feature_importance_torch.json"), "w") as f:
        json.dump(imp, f, indent=2)
    V.plot_feature_importance(imp, fname="feature_importance_torch.png")
    print("      Najvaznije znacajke:")
    for name, val in list(imp.items())[:10]:
        print(f"        {name:28s}: {val:.4f}")

    model_path = os.path.join(C.MODELS_DIR, "transformer_torch.pt")
    torch.save({"state_dict": last_model.state_dict(), "params": PARAMS,
                "seq_len": SEQ_LEN, "feat_cols": feat_cols}, model_path)
    print(f"      Model spremljen: {model_path}")

    print("\n" + "=" * 64)
    print(" GOTOVO. Prosjecni rezultati u results/metrics_torch_cv.json")
    print("=" * 64)

if __name__ == "__main__":
    main()
