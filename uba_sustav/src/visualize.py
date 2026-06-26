import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

sns.set_theme(style="whitegrid")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_roc_curves(y_true, score_dict, filename="roc_krivulje.png"):
    plt.figure(figsize=(8, 6))
    for name, scores in score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Slucajno pogadjanje")
    plt.xlabel("Stopa laznih pozitiva (FPR)")
    plt.ylabel("Stopa istinitih pozitiva (TPR)")
    plt.title("ROC krivulje - usporedba modela detekcije anomalija")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_metrics_bar(metrics_df, filename="usporedba_metrika.png"):
    plot_df = metrics_df[["precision", "recall", "f1", "roc_auc"]].reset_index()
    plot_df = plot_df.melt(id_vars="model", var_name="metrika", value_name="vrijednost")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="metrika", y="vrijednost", hue="model")
    plt.title("Usporedba modela po metrikama")
    plt.ylim(0, 1.05)
    plt.ylabel("Vrijednost")
    plt.xlabel("Metrika")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_score_distribution(y_true, scores, name="Isolation Forest",
                            filename="distribucija_scoreova.png"):
    plt.figure(figsize=(9, 5))
    df = pd.DataFrame({"score": scores, "klasa": np.where(y_true == 1, "Zlonamjerno", "Normalno")})
    sns.histplot(data=df, x="score", hue="klasa", bins=50, kde=True,
                 palette={"Normalno": "#4C72B0", "Zlonamjerno": "#C44E52"})
    plt.title(f"Distribucija anomaly scoreova - {name}")
    plt.xlabel("Anomaly score (vise = anomalnije)")
    plt.ylabel("Broj uzoraka (user-dan)")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_confusion(y_true, preds, name="Isolation Forest",
                   filename="matrica_konfuzije.png"):
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normalno", "Zlonamjerno"],
                yticklabels=["Normalno", "Zlonamjerno"])
    plt.title(f"Matrica konfuzije - {name}")
    plt.ylabel("Stvarna klasa")
    plt.xlabel("Predvidjena klasa")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def top_anomalies(features_df, scores, n=15):
    df = features_df.copy()
    df["anomaly_score"] = scores
    cols = ["user", "day", "anomaly_score", "num_after_hours_logons",
            "num_usb_connects", "num_files_to_usb", "num_external_emails",
            "num_suspicious_http", "malicious"]
    return df.sort_values("anomaly_score", ascending=False)[cols].head(n)
