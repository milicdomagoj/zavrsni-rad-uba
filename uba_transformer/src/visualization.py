import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from . import config as C

def plot_class_distribution(daily_df, fname="eda_class_distribution.png"):
    counts = daily_df["malicious"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["Normalno", "Zlonamjerno"], [counts.get(0, 0), counts.get(1, 0)],
            color=["#4C72B0", "#C44E52"])
    plt.yscale("log")
    plt.ylabel("Broj (korisnik-dan) uzoraka  [log]")
    plt.title("Raspodjela klasa (neuravnotezenost)")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        plt.text(i, v, f"{v:,}", ha="center", va="bottom")
    plt.tight_layout(); _save(fname)

def plot_roc(y, probs, fname="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC krivulja (Transformer)")
    plt.legend(); plt.tight_layout(); _save(fname)

def plot_pr(y, probs, fname="pr_curve.png"):
    prec, rec, _ = precision_recall_curve(y, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, lw=2, label=f"PR-AUC = {auc(rec, prec):.3f}")
    base = y.mean()
    plt.axhline(base, ls="--", c="gray", label=f"slucajno = {base:.3f}")
    plt.xlabel("Odziv"); plt.ylabel("Preciznost"); plt.title("Precision-Recall krivulja")
    plt.legend(); plt.tight_layout(); _save(fname)

def plot_learning_curves(history, fname="learning_curves.png"):
    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.get("loss", []), label="trening")
    ax[0].plot(h.get("val_loss", []), label="validacija")
    ax[0].set_title("Gubitak"); ax[0].set_xlabel("epoha"); ax[0].legend()
    key = "pr_auc" if "pr_auc" in h else "auc"
    ax[1].plot(h.get(key, []), label="trening")
    ax[1].plot(h.get("val_" + key, []), label="validacija")
    ax[1].set_title(key.upper()); ax[1].set_xlabel("epoha"); ax[1].legend()
    plt.tight_layout(); _save(fname)

def plot_confusion(cm, fname="confusion_matrix.png"):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["Normalno", "Zlonamjerno"])
    plt.yticks([0, 1], ["Normalno", "Zlonamjerno"])
    plt.xlabel("Predvidjeno"); plt.ylabel("Stvarno"); plt.title("Matrica konfuzije")
    plt.tight_layout(); _save(fname)

def plot_feature_importance(importances, top=20, fname="feature_importance.png"):
    items = list(importances.items())[:top]
    names = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    plt.figure(figsize=(8, max(4, 0.32 * len(names))))
    plt.barh(names, vals, color="#2c6fbb")
    plt.xlabel("Pad PR-AUC kad se znacajka izmijesa")
    plt.title("Permutacijska vaznost znacajki (Top %d)" % top)
    plt.tight_layout(); _save(fname)

def plot_attention(per_step, fname="attention_per_step.png"):
    plt.figure(figsize=(9, 3.5))
    plt.plot(per_step, color="#c0392b")
    plt.xlabel("Vremenski korak u sekvenci (dan)")
    plt.ylabel("Prosjecna paznja")
    plt.title("Koje dane u sekvenci model najvise 'gleda'")
    plt.tight_layout(); _save(fname)

def _save(fname):
    path = os.path.join(C.RESULTS_DIR, fname)
    plt.savefig(path, dpi=150); plt.close()
    print(f"  spremljeno: {path}")
