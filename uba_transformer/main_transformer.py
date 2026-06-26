import os
import sys
import json
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as C
from src import data_loading as D
from src import feature_engineering as FE
from src import sequences as S
from src import training as T
from src import visualization as V
from src.transformer_model import build_transformer

def main():
    print("=" * 64)
    print(" INSAJDERSKA PRIJETNJA - TRANSFORMER ENCODER (CERT r4.2)")
    print("=" * 64)
    np.random.seed(C.SEED); tf.random.set_seed(C.SEED)
    T.enable_mixed_precision()

    print("\n[1/9] Eksploratorna analiza (EDA)...")
    D.explore_dataset(verbose=True)

    print("\n[2/9] Izgradnja dnevnih znacajki...")
    df = FE.build_daily_features(use_cache=True, verbose=True)
    feat_cols = FE.feature_columns(df)
    print(f"      Znacajki: {len(feat_cols)}")
    dist = D.class_distribution(df)
    print(f"      Raspodjela klasa: {dist}")
    V.plot_class_distribution(df)

    print("\n[3/9] Odabir najbolje duljine sekvence (30/50/100/200)...")
    best_L, seq_results = T.select_best_seq_length(df, feat_cols)

    print(f"\n[4/9] Izgradnja sljedova (L={best_L}) + podjela po korisniku...")
    X, y, groups = S.make_sequences(df, feat_cols, seq_len=best_L)
    tr, va, te = S.split_by_user(groups, y)
    X, scaler = S.scale_sequences(X, tr, len(feat_cols))
    print(f"      X={X.shape} | trening={tr.sum()} val={va.sum()} test={te.sum()}")
    print(f"      pozitivnih: trening={int(y[tr].sum())} val={int(y[va].sum())} test={int(y[te].sum())}")

    print(f"\n[5/9] Optuna optimizacija hiperparametara ({C.OPTUNA_TRIALS} pokusaja)...")
    best_params, study = T.optimize_hyperparams(X, y, tr, va, len(feat_cols), best_L)
    with open(os.path.join(C.RESULTS_DIR, "best_params.json"), "w") as f:
        json.dump({"seq_len": int(best_L), "params": best_params,
                   "seq_results": {str(k): float(v) for k, v in seq_results.items()}}, f, indent=2)

    print("\n[6/9] Finalno treniranje s najboljim hiperparametrima...")
    sel = S.subsample_negatives(tr, y)
    cw = S.compute_class_weight(y[sel])
    model = build_transformer(best_L, len(feat_cols), **best_params)
    model_path = os.path.join(C.MODELS_DIR, "transformer_best.keras")
    history = model.fit(X[sel], y[sel], validation_data=(X[va], y[va]),
                        epochs=C.MAX_EPOCHS, batch_size=C.BATCH_SIZE,
                        class_weight=cw, callbacks=T.make_callbacks(model_path), verbose=2)
    V.plot_learning_curves(history)

    print("\n[7/9] Evaluacija na test skupu...")
    metrics, cm, probs = T.evaluate(model, X[te], y[te])
    print("      Rezultati (test):")
    for k, v in metrics.items():
        print(f"        {k:12s}: {v:.4f}")
    with open(os.path.join(C.RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    V.plot_roc(y[te], probs); V.plot_pr(y[te], probs); V.plot_confusion(cm)

    print("\n[8/9] Interpretabilnost (vaznost znacajki + paznja)...")
    importances = T.permutation_importance(model, X[te], y[te], feat_cols, n_repeats=2)
    with open(os.path.join(C.RESULTS_DIR, "feature_importance.json"), "w") as f:
        json.dump(importances, f, indent=2)
    V.plot_feature_importance(importances)
    print("      Najvaznije znacajke:")
    for name, val in list(importances.items())[:10]:
        print(f"        {name:28s}: {val:.4f}")
    per_step = T.attention_summary(model, X[te][:512])
    if per_step is not None:
        V.plot_attention(per_step)

    print("\n[9/9] Spremanje modela...")
    model.save(model_path)
    print(f"      Model spremljen: {model_path}")

    print("\n" + "=" * 64)
    print(" GOTOVO. Rezultati i grafovi u results/, model u models/")
    print("=" * 64)

if __name__ == "__main__":
    main()
