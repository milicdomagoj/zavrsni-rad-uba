import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

RANDOM_SEED = 42

def prepare_matrix(features_df, feat_cols):
    X = features_df[feat_cols].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = features_df["malicious"].values
    return X_scaled, y, scaler

def _scores_to_labels(scores, contamination):
    threshold = np.quantile(scores, 1 - contamination)
    return (scores >= threshold).astype(int)

def run_isolation_forest(X, contamination=0.05):
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=RANDOM_SEED)
    model.fit(X)
    scores = -model.decision_function(X)
    preds = (model.predict(X) == -1).astype(int)
    return scores, preds

def run_ocsvm(X, contamination=0.05):
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=contamination)
    model.fit(X)
    scores = -model.decision_function(X)
    preds = (model.predict(X) == -1).astype(int)
    return scores, preds

def run_lof(X, contamination=0.05):
    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    preds = (model.fit_predict(X) == -1).astype(int)
    scores = -model.negative_outlier_factor_
    return scores, preds

def run_baseline(features_df, feat_cols, contamination=0.05):
    col = "num_files_to_usb"
    scores = features_df[col].values.astype(float)
    preds = _scores_to_labels(scores, contamination)
    return scores, preds

def evaluate(y_true, y_pred, scores, name):
    metrics = {
        "model": name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else float("nan"),
    }
    return metrics

def run_all_models(features_df, feat_cols, contamination=0.05):
    X, y, _ = prepare_matrix(features_df, feat_cols)

    results = []
    score_dict = {}

    s, p = run_isolation_forest(X, contamination)
    results.append(evaluate(y, p, s, "Isolation Forest"))
    score_dict["Isolation Forest"] = s

    s, p = run_ocsvm(X, contamination)
    results.append(evaluate(y, p, s, "One-Class SVM"))
    score_dict["One-Class SVM"] = s

    s, p = run_lof(X, contamination)
    results.append(evaluate(y, p, s, "Local Outlier Factor"))
    score_dict["Local Outlier Factor"] = s

    s, p = run_baseline(features_df, feat_cols, contamination)
    results.append(evaluate(y, p, s, "Baseline (prag USB)"))
    score_dict["Baseline (prag USB)"] = s

    metrics_df = pd.DataFrame(results).set_index("model")
    return metrics_df, score_dict, y

if __name__ == "__main__":
    print("model.py se koristi kao modul (pozovite ga iz main_cert_full.py).")
