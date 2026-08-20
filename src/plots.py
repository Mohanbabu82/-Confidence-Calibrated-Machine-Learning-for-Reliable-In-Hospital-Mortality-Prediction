"""Plotting utilities: reliability diagrams, ROC/PR curves, and cohort-audit plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

sns.set_theme(style="whitegrid")


def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, save_path: str | Path | None = None):
    """Plot a reliability (calibration) diagram comparing predicted vs. observed frequency."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Diagram")
    ax.legend()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig


def plot_reliability_diagram_with_histogram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """Two-panel calibration plot: reliability curve on top, a histogram of
    predicted-probability counts per bin (the "confidence histogram") below,
    sharing the x-axis. Bin edges are handled internally to avoid sklearn's
    calibration_curve silently dropping empty bins, so the histogram panel
    always reflects the same n_bins as the reliability curve above it.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    bin_acc = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        bin_count[i] = mask.sum()
        if mask.sum() > 0:
            bin_acc[i] = y_true[mask].mean()

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(6, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_top.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax_top.bar(
        bin_centers, np.nan_to_num(bin_acc), width=bin_width * 0.9,
        edgecolor="black", color="steelblue", alpha=0.7, label="Observed frequency",
    )
    ax_top.set_ylabel("Observed frequency")
    ax_top.set_ylim(0, 1)
    ax_top.set_xlim(0, 1)
    ax_top.legend(loc="upper left")
    ax_top.set_title(title or "Reliability Diagram")

    ax_bottom.bar(bin_centers, bin_count, width=bin_width * 0.9, color="steelblue", edgecolor="black")
    ax_bottom.set_xlabel("Predicted probability")
    ax_bottom.set_ylabel("Count")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str | Path | None = None):
    """Plot the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str | Path | None = None):
    """Plot the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label="Model")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig


def plot_missingness(df: pd.DataFrame, save_path: str | Path | None = None, top_n: int | None = None):
    """Bar chart of percent-missing per column, sorted descending.

    Columns with zero missing values are omitted so the plot stays
    readable on wide cohort tables.
    """
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    if top_n is not None:
        missing_pct = missing_pct.head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(missing_pct))))
    if len(missing_pct) == 0:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        sns.barplot(x=missing_pct.values, y=missing_pct.index, ax=ax, color="steelblue")
        ax.set_xlabel("Missing (%)")
        ax.set_ylabel("")
    ax.set_title("Missing Data by Column")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig


def plot_outcome_balance(
    df: pd.DataFrame,
    target_col: str,
    group_col: str | None = None,
    save_path: str | Path | None = None,
):
    """Bar chart of outcome class counts, optionally split by a grouping
    column (e.g. train/val/test) to visually confirm stratification held.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    if group_col is None:
        counts = df[target_col].value_counts().sort_index()
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="steelblue")
        ax.set_xlabel(target_col)
        ax.set_ylabel("Count")
    else:
        plot_df = df[[group_col, target_col]].copy()
        sns.countplot(data=plot_df, x=group_col, hue=target_col, ax=ax)
        ax.set_xlabel(group_col)
        ax.set_ylabel("Count")
        ax.legend(title=target_col)

    ax.set_title("Outcome Class Balance")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig


def _plot_metric_vs_coverage(
    policy_df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    model_col: str = "model",
    coverage_col: str = "coverage",
    save_path: str | Path | None = None,
):
    """Shared line-plot helper for selective-prediction coverage curves:
    one line per model, x=coverage (descending as referral rate rises),
    y=the requested metric column.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, group in policy_df.groupby(model_col):
        group_sorted = group.sort_values(coverage_col)
        ax.plot(group_sorted[coverage_col], group_sorted[y_col], marker="o", label=model_name)

    ax.set_xlabel("Coverage (fraction of cases NOT referred to clinician review)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.invert_xaxis()  # coverage decreases left-to-right as referral rate increases
    ax.legend(title="Model")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig


def plot_risk_coverage_curve(policy_df: pd.DataFrame, save_path: str | Path | None = None):
    """Risk-coverage curve: selective risk (accepted-case error rate) vs.
    coverage, one line per model. Selective risk should fall as coverage
    drops (i.e. as more uncertain cases are referred to clinician review).
    """
    return _plot_metric_vs_coverage(
        policy_df, y_col="selective_risk", y_label="Selective risk (accepted-case error rate)",
        title="Risk-Coverage Curve", save_path=save_path,
    )


def plot_accuracy_coverage_curve(policy_df: pd.DataFrame, save_path: str | Path | None = None):
    """Accuracy-coverage curve: 1 - selective risk vs. coverage."""
    plot_df = policy_df.copy()
    plot_df["accepted_accuracy"] = 1 - plot_df["selective_risk"]
    return _plot_metric_vs_coverage(
        plot_df, y_col="accepted_accuracy", y_label="Accuracy on accepted cases",
        title="Accuracy-Coverage Curve", save_path=save_path,
    )


def plot_calibration_vs_coverage_curve(policy_df: pd.DataFrame, save_path: str | Path | None = None):
    """Calibration-vs-coverage curve: ECE (10 bins) on the accepted subset
    vs. coverage, one line per model.
    """
    return _plot_metric_vs_coverage(
        policy_df, y_col="ece", y_label="Expected Calibration Error (10 bins, accepted subset)",
        title="Calibration-vs-Coverage Curve", save_path=save_path,
    )
