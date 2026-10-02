"""STAGE 9 - EVALUATION: metrics, confusion matrix, disaggregated reports, error analysis."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from .utils import FIGURES_DIR


def metrics(y_true, y_pred, labels=None) -> dict:
    kw = dict(labels=labels, zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", **kw),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "macro_precision": precision_score(y_true, y_pred, average="macro", **kw),
            "macro_recall": recall_score(y_true, y_pred, average="macro", **kw),
            "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0)}


def per_class_report(y_true, y_pred, labels) -> pd.DataFrame:
    return pd.DataFrame(classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)).T


def disaggregated(df_test: pd.DataFrame, y_pred) -> pd.DataFrame:
    """Original-dataset intents vs synthetic (enriched) intents, reported separately."""
    rows = []
    d = df_test.assign(pred=y_pred)
    for name, sub in (("ALL", d), ("original intents", d[d.source == "original"]),
                      ("enriched intents (synthetic)", d[d.source == "synthetic"])):
        m = metrics(sub.intent, sub.pred, labels=sorted(sub.intent.unique())); m.update(view=name, n=len(sub)); rows.append(m)
    return pd.DataFrame(rows).set_index("view")


def plot_confusion(y_true, y_pred, labels, path=None, title="Confusion Matrix (held-out test set)"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.xticks(rotation=40, ha="right"); plt.tight_layout()
    fig.savefig(path or FIGURES_DIR / "confusion_matrix.png", dpi=160); plt.close(fig)
    return cm


def error_analysis(df_test, y_pred, proba, classes) -> pd.DataFrame:
    d = df_test.assign(predicted=y_pred, confidence=proba.max(axis=1).round(3))
    err = d[d.intent != d.predicted][["query", "intent", "predicted", "confidence", "source"]]
    return err.rename(columns={"intent": "true_intent"}).sort_values("confidence", ascending=False)


def confusion_pairs(err: pd.DataFrame) -> pd.DataFrame:
    return err.groupby(["true_intent", "predicted"]).size().sort_values(ascending=False).rename("count").reset_index()


def plot_confidence(y_true, y_pred, proba, label_names):
    conf = proba.max(axis=1); ok = np.array(y_true) == np.array(y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].hist([conf[ok], conf[~ok]], bins=15, label=["correct", "wrong"], color=["#2a9d8f", "#e76f51"], stacked=True)
    ax[0].set_title("Confidence: correct vs wrong"); ax[0].set_xlabel("max probability"); ax[0].legend()
    frac, mean = calibration_curve(ok.astype(int), conf, n_bins=8, strategy="quantile")
    ax[1].plot(mean, frac, "o-", label="model"); ax[1].plot([0, 1], [0, 1], "k--", label="perfect")
    ax[1].set_title("Calibration curve (top-1 confidence)"); ax[1].set_xlabel("mean confidence"); ax[1].set_ylabel("accuracy"); ax[1].legend()
    plt.tight_layout(); fig.savefig(FIGURES_DIR / "confidence_calibration.png", dpi=150); plt.close(fig)
