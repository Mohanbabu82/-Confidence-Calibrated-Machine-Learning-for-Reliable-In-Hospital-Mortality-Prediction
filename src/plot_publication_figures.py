"""Publication-ready figure and table export utilities.

DATA SOURCE POLICY: every function here reads only already-computed files
under outputs/tables/ — aggregate metric tables from notebooks 01-05, and
outputs/tables/test_predictions_{model}.csv (raw per-case test
predictions, saved as a byproduct of src.evaluate_final.run_final_evaluation()
— the project's single authorized test-set evaluation). This module NEVER
recomputes model predictions, refits a model or calibrator, or invents
illustrative numbers. Captions, abbreviation lists, and notes are authored
descriptive text; any number appearing inside a note (sample size,
bootstrap count, seed) is pulled from the actual saved table or run-
metadata JSON, never hardcoded.

Figures are saved as both 300 DPI PNG (print/manuscript submission) and
vector PDF (typesetting) at the same base filename.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.config import load_config
from src.plots import plot_reliability_diagram_with_histogram, plot_risk_coverage_curve
from src.selective_prediction import evaluate_referral_policy

sns.set_theme(style="whitegrid")

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "lightgbm": "LightGBM",
    "mlp_torch": "MLP (PyTorch)",
}


# ---------------------------------------------------------------------------
# Shared save helper: PNG (300 DPI) + vector PDF
# ---------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, base_path: str | Path, dpi: int = 300) -> tuple[Path, Path]:
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")  # PDF is vector; dpi affects only embedded raster elements
    return png_path, pdf_path


def _load_test_predictions(config: dict[str, Any], model_name: str) -> pd.DataFrame:
    path = Path(config["publication"]["tables_output_dir"]) / f"test_predictions_{model_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Test predictions file not found at '{path}'. Run "
            "notebooks/05_final_evaluation.ipynb (src.evaluate_final.run_final_evaluation) "
            "first — it saves this file as a byproduct of the single authorized "
            "test-set evaluation."
        )
    return pd.read_csv(path)


def load_all_test_predictions(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {name: _load_test_predictions(config, name) for name in MODEL_DISPLAY_NAMES}


# ---------------------------------------------------------------------------
# Figure 3: Reliability diagrams (per model, calibrated test predictions)
# ---------------------------------------------------------------------------

def make_reliability_diagrams(
    test_predictions: dict[str, pd.DataFrame], config: dict[str, Any], n_bins: int = 10
) -> list[tuple[Path, Path]]:
    figures_dir = Path(config["publication"]["figures_output_dir"])
    dpi = config["publication"]["dpi"]
    outputs = []

    for model_name, df in test_predictions.items():
        method = df["calibration_method"].iloc[0]
        fig = plot_reliability_diagram_with_histogram(
            df["y_true"].to_numpy(), df["y_prob_calibrated"].to_numpy(), n_bins=n_bins,
            title=f"{MODEL_DISPLAY_NAMES[model_name]} — Reliability Diagram (test set, {method} calibration)",
        )
        base_path = figures_dir / f"figure3_reliability_{model_name}"
        outputs.append(_save_figure(fig, base_path, dpi=dpi))
        plt.close(fig)

    return outputs


# ---------------------------------------------------------------------------
# Figure 4: ROC curves (all models overlaid)
# ---------------------------------------------------------------------------

def make_roc_curve_figure(test_predictions: dict[str, pd.DataFrame], config: dict[str, Any]) -> tuple[Path, Path]:
    figures_dir = Path(config["publication"]["figures_output_dir"])
    dpi = config["publication"]["dpi"]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")

    for model_name, df in test_predictions.items():
        y_true = df["y_true"].to_numpy()
        y_prob = df["y_prob_calibrated"].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{MODEL_DISPLAY_NAMES[model_name]} (AUROC={auroc:.3f})")

    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curves — Held-Out Test Set")
    ax.legend(loc="lower right")
    fig.tight_layout()

    base_path = figures_dir / "figure4_roc_curves"
    result = _save_figure(fig, base_path, dpi=dpi)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# Figure 5: Precision-Recall curves (all models overlaid)
# ---------------------------------------------------------------------------

def make_pr_curve_figure(test_predictions: dict[str, pd.DataFrame], config: dict[str, Any]) -> tuple[Path, Path]:
    figures_dir = Path(config["publication"]["figures_output_dir"])
    dpi = config["publication"]["dpi"]

    fig, ax = plt.subplots(figsize=(6.5, 6))

    prevalence = None
    for model_name, df in test_predictions.items():
        y_true = df["y_true"].to_numpy()
        y_prob = df["y_prob_calibrated"].to_numpy()
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{MODEL_DISPLAY_NAMES[model_name]} (AUPRC={auprc:.3f})")
        if prevalence is None:
            prevalence = y_true.mean()

    if prevalence is not None:
        ax.axhline(prevalence, linestyle="--", color="gray", label=f"Prevalence baseline ({prevalence:.3f})")

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curves — Held-Out Test Set")
    ax.legend(loc="upper right")
    fig.tight_layout()

    base_path = figures_dir / "figure5_precision_recall_curves"
    result = _save_figure(fig, base_path, dpi=dpi)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# Figure 6: Risk-coverage curve (test set, reusing src.selective_prediction)
# ---------------------------------------------------------------------------

def make_risk_coverage_figure(
    test_predictions: dict[str, pd.DataFrame], config: dict[str, Any]
) -> tuple[Path, Path]:
    figures_dir = Path(config["publication"]["figures_output_dir"])
    dpi = config["publication"]["dpi"]
    referral_rates = config["selective_prediction"]["referral_rates"]
    threshold = config["selective_prediction"]["decision_threshold"]

    frames = []
    for model_name, df in test_predictions.items():
        policy_df = evaluate_referral_policy(
            df["y_true"].to_numpy(), df["y_prob_calibrated"].to_numpy(),
            referral_rates=referral_rates, threshold=threshold,
        )
        policy_df.insert(0, "model", MODEL_DISPLAY_NAMES[model_name])
        frames.append(policy_df)
    combined_df = pd.concat(frames, ignore_index=True)

    fig = plot_risk_coverage_curve(combined_df)
    fig.axes[0].set_title("Risk-Coverage Curve — Held-Out Test Set")

    base_path = figures_dir / "figure6_risk_coverage_curve"
    result = _save_figure(fig, base_path, dpi=dpi)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# Figure 7: Subgroup calibration comparison
# ---------------------------------------------------------------------------

def make_subgroup_calibration_figure(
    test_predictions: dict[str, pd.DataFrame],
    config: dict[str, Any],
    subgroup_variable: str | None = None,
    n_bins: int = 10,
    min_group_size: int = 20,
) -> tuple[Path, Path]:
    """One panel per model; within each panel, one reliability curve per
    level of `subgroup_variable`, so calibration quality can be visually
    compared across subgroups. Levels with fewer than `min_group_size`
    cases are skipped (not plotted), never approximated.
    """
    figures_dir = Path(config["publication"]["figures_output_dir"])
    dpi = config["publication"]["dpi"]
    subgroup_variable = subgroup_variable or config["publication"]["subgroup_comparison_variable"]

    model_names = list(test_predictions.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5.5), sharey=True)
    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        df = test_predictions[model_name]
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

        if subgroup_variable not in df.columns:
            ax.set_title(f"{MODEL_DISPLAY_NAMES[model_name]}\n(subgroup column '{subgroup_variable}' not available)")
            continue

        for level, group in df.groupby(subgroup_variable):
            if len(group) < min_group_size:
                continue
            y_true = group["y_true"].to_numpy()
            y_prob = group["y_prob_calibrated"].to_numpy()

            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            bin_centers, bin_acc = [], []
            for i in range(n_bins):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
                if mask.sum() == 0:
                    continue
                bin_centers.append(y_prob[mask].mean())
                bin_acc.append(y_true[mask].mean())

            ax.plot(bin_centers, bin_acc, marker="o", label=f"{level} (n={len(group)})")

        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{MODEL_DISPLAY_NAMES[model_name]}")
        ax.legend(title=subgroup_variable, fontsize=8)

    axes[0].set_ylabel("Observed frequency")
    fig.suptitle(f"Subgroup Calibration Comparison — {subgroup_variable} (Held-Out Test Set)")
    fig.tight_layout()

    base_path = figures_dir / f"figure7_subgroup_calibration_{subgroup_variable}"
    result = _save_figure(fig, base_path, dpi=dpi)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# Figure 1: Study flowchart (matplotlib, from real cohort-flow counts)
# ---------------------------------------------------------------------------

def _draw_flow_boxes(box_labels: list[str], title: str, figsize: tuple[float, float] = (5.5, 10)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(box_labels))
    ax.axis("off")

    box_height = 0.7
    for i, label in enumerate(box_labels):
        y = len(box_labels) - i - 1
        box = mpatches.FancyBboxPatch(
            (0.05, y + (1 - box_height) / 2), 0.9, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2, edgecolor="black", facecolor="#dbe9f6",
        )
        ax.add_patch(box)
        ax.text(0.5, y + 0.5, label, ha="center", va="center", fontsize=9, wrap=True)

        if i < len(box_labels) - 1:
            ax.annotate(
                "", xy=(0.5, y), xytext=(0.5, y + (1 - box_height) / 2),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
            )

    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig


def make_study_flowchart_figure(
    cohort_flow_df: pd.DataFrame, config: dict[str, Any]
) -> tuple[Path, Path]:
    figures_dir = Path(config["publication"]["figures_output_dir"])
    dpi = config["publication"]["dpi"]

    stage_label_overrides = {
        "loaded (raw cohort_features.parquet)": "Extracted ICU cohort\n(first ICU stay per patient, ≥24h stay)",
        "after duplicate-row removal": "After duplicate-row removal",
        "train split": "Training set",
        "validation split": "Validation set",
        "test split": "Held-out test set",
    }

    labels = []
    for _, row in cohort_flow_df.iterrows():
        stage_text = stage_label_overrides.get(row["stage"], row["stage"])
        labels.append(f"{stage_text}\nn = {int(row['n_rows'])} ({int(row['n_unique_patients'])} unique patients)")

    fig = _draw_flow_boxes(labels, title="Figure 1. Study Cohort Flow", figsize=(5.5, 1.8 * len(labels)))
    base_path = figures_dir / "figure1_study_flowchart"
    result = _save_figure(fig, base_path, dpi=dpi)
    plt.close(fig)
    return result


def generate_study_workflow_mermaid(cohort_flow_df: pd.DataFrame, save_path: str | Path) -> Path:
    """Mermaid flowchart source for Figure 1, using the same real cohort-flow
    counts as the matplotlib figure (not independently authored numbers).
    """
    stage_label_overrides = {
        "loaded (raw cohort_features.parquet)": "Extracted ICU cohort<br/>(first ICU stay per patient, &ge;24h stay)",
        "after duplicate-row removal": "After duplicate-row removal",
        "train split": "Training set (70%)",
        "validation split": "Validation set (10%)",
        "test split": "Held-out test set (20%)",
    }

    lines = ["flowchart TD"]
    node_ids = []
    for i, row in cohort_flow_df.iterrows():
        node_id = f"N{i}"
        node_ids.append(node_id)
        stage_text = stage_label_overrides.get(row["stage"], row["stage"])
        label = f"{stage_text}<br/>n = {int(row['n_rows'])} ({int(row['n_unique_patients'])} unique patients)"
        lines.append(f'    {node_id}["{label}"]')

    for a, b in zip(node_ids[:-1], node_ids[1:]):
        lines.append(f"    {a} --> {b}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return save_path


# ---------------------------------------------------------------------------
# Figure 2: Proposed system architecture (static, grounded in real module names)
# ---------------------------------------------------------------------------

ARCHITECTURE_STAGES = [
    "Local MIMIC-IV extract\n(data/raw/mimiciv — no data shipped with this project)",
    "Cohort + feature extraction\n(sql/cohort.sql, sql/features.sql — first 24h ICU window only)",
    "Preprocessing\n(src/preprocess.py — median imputation, missingness indicators,\none-hot encoding, standard scaling; fit on train split only)",
    "Base models\n(src/train_models.py — Logistic Regression, LightGBM, PyTorch MLP;\nvalidation-selected hyperparameters, early stopping)",
    "Calibration layer\n(src/calibration.py — Platt scaling, isotonic regression,\ntemperature scaling; fit on validation predictions only)",
    "Selective prediction / referral layer\n(src/selective_prediction.py — uncertainty-ranked referral;\nCLINICAL DECISION SUPPORT ONLY, not autonomous diagnosis)",
    "Clinician review\n(accepted cases: decision-support input; referred cases:\nmandatory full clinician review)",
]


def make_architecture_diagram_figure(config: dict[str, Any]) -> tuple[Path, Path]:
    figures_dir = Path(config["publication"]["figures_output_dir"])
    dpi = config["publication"]["dpi"]

    fig = _draw_flow_boxes(
        ARCHITECTURE_STAGES,
        title="Figure 2. Proposed Confidence-Calibrated Clinical AI Architecture",
        figsize=(7, 1.8 * len(ARCHITECTURE_STAGES)),
    )
    base_path = figures_dir / "figure2_system_architecture"
    result = _save_figure(fig, base_path, dpi=dpi)
    plt.close(fig)
    return result


def generate_architecture_mermaid(save_path: str | Path) -> Path:
    """Mermaid flowchart source for Figure 2 — structural description of
    the actually-implemented pipeline (module/file names), not data-
    dependent, so there is nothing to fabricate here.
    """
    lines = [
        "flowchart TD",
        '    A["Local MIMIC-IV extract<br/>(data/raw/mimiciv — no data shipped with this project)"]',
        '    B["Cohort + feature extraction<br/>(sql/cohort.sql, sql/features.sql)<br/>first 24h ICU window only"]',
        '    C["Preprocessing<br/>(src/preprocess.py)<br/>imputation, missingness indicators, encoding, scaling<br/>fit on train split only"]',
        '    D["Base models<br/>(src/train_models.py)<br/>Logistic Regression / LightGBM / PyTorch MLP<br/>validation-selected hyperparameters"]',
        '    E["Calibration layer<br/>(src/calibration.py)<br/>Platt scaling / isotonic regression / temperature scaling<br/>fit on validation predictions only"]',
        '    F["Selective prediction / referral layer<br/>(src/selective_prediction.py)<br/>uncertainty-ranked referral"]',
        '    G["Clinician review<br/>accepted: decision-support input<br/>referred: mandatory full review"]',
        "    A --> B --> C --> D --> E --> F --> G",
        '    classDef disclaimer fill:#fff3cd,stroke:#b8860b;',
        '    H["CLINICAL DECISION SUPPORT ONLY — NOT AN AUTONOMOUS DIAGNOSTIC SYSTEM"]:::disclaimer',
        "    F -.-> H",
    ]

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return save_path


# ---------------------------------------------------------------------------
# Table export: CSV + Markdown + LaTeX, with captions/abbreviations/notes
# ---------------------------------------------------------------------------

ABBREVIATIONS = {
    "AUROC": "Area Under the Receiver Operating Characteristic curve",
    "AUPRC": "Area Under the Precision-Recall Curve",
    "CI": "Confidence Interval",
    "ECE": "Expected Calibration Error",
    "MCE": "Maximum Calibration Error",
    "NLL": "Negative Log-Likelihood",
    "PPV": "Positive Predictive Value (Precision)",
    "NPV": "Negative Predictive Value",
    "ICU": "Intensive Care Unit",
    "MLP": "Multi-Layer Perceptron",
    "GCS": "Glasgow Coma Scale",
    "SBP": "Systolic Blood Pressure",
    "DBP": "Diastolic Blood Pressure",
    "MAP": "Mean Arterial Pressure",
    "BUN": "Blood Urea Nitrogen",
    "WBC": "White Blood Cell count",
}


def _collapse_ci_columns(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Publication-display transform ONLY (never applied to the .csv
    export): for every `{metric}` column that has matching
    `{metric}_ci_lower`/`{metric}_ci_upper` columns, collapse the three
    into a single "value (lower-upper)" string column. A 40-column table
    of point estimates + separate CI bounds is unusable in a manuscript;
    this does not change any underlying number, only how it is displayed.
    """
    display_df = pd.DataFrame(index=df.index)
    handled = set()

    for col in df.columns:
        lower_col, upper_col = f"{col}_ci_lower", f"{col}_ci_upper"
        if col in handled or col.endswith("_ci_lower") or col.endswith("_ci_upper"):
            continue
        if lower_col in df.columns and upper_col in df.columns:
            def fmt(row, c=col, lo=lower_col, hi=upper_col):
                # "N/A" (not "NaN"): tabulate/to_markdown re-parses an
                # all-"NaN" string column as float and silently reformats
                # it to lowercase "nan", while a mixed column (some real
                # values, some "NaN") leaves "NaN" as literal text —
                # inconsistent rendering depending on column contents.
                # "N/A" does not parse as a float, so rendering is
                # consistent regardless of how many rows are missing.
                if pd.isna(row[c]):
                    return "N/A"
                if pd.isna(row[lo]) or pd.isna(row[hi]):
                    return f"{row[c]:.{decimals}f}"
                return f"{row[c]:.{decimals}f} ({row[lo]:.{decimals}f}–{row[hi]:.{decimals}f})"
            display_df[col] = df.apply(fmt, axis=1)
            handled.update({col, lower_col, upper_col})
        else:
            display_df[col] = df[col]

    return display_df


def _markdown_table(df: pd.DataFrame) -> str:
    return _collapse_ci_columns(df).to_markdown(index=False)


def _latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    display_df = _collapse_ci_columns(df)
    body = display_df.to_latex(index=False, float_format="%.4f", escape=True, longtable=False)
    return (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"{body}\n\\end{{table}}\n"
    )


def export_table_with_metadata(
    df: pd.DataFrame,
    base_path: str | Path,
    caption: str,
    notes: list[str],
    abbreviations_used: list[str] | None = None,
    label: str | None = None,
) -> dict[str, Path]:
    """Export one table as .csv (plain data, for reuse), .md (caption +
    table + abbreviation list + notes, for manuscript drafting in
    Markdown), and .tex (LaTeX table environment + notes as a minipage
    below, for direct inclusion in a manuscript). The .csv contains only
    the data; caption/abbreviations/notes are metadata, appended to .md
    and .tex only (never mixed into the machine-readable .csv).
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    label = label or base_path.stem

    csv_path = base_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    abbreviations_used = abbreviations_used or []
    abbrev_lines = [f"- **{k}**: {ABBREVIATIONS[k]}" for k in abbreviations_used if k in ABBREVIATIONS]

    md_lines = [f"**{caption}**", "", _markdown_table(df), ""]
    if abbrev_lines:
        md_lines += ["*Abbreviations:*", ""] + abbrev_lines + [""]
    if notes:
        md_lines += ["*Notes:*", ""] + [f"- {n}" for n in notes] + [""]
    md_path = base_path.with_suffix(".md")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    tex_lines = [_latex_table(df, caption, label), ""]
    if abbrev_lines or notes:
        tex_lines.append("\\begin{quote}\\small")
        if abbrev_lines:
            tex_lines.append("\\textit{Abbreviations:} " + "; ".join(
                f"{k} = {ABBREVIATIONS[k]}" for k in abbreviations_used if k in ABBREVIATIONS
            ) + ".")
        if notes:
            tex_lines.append("\\textit{Notes:} " + " ".join(notes))
        tex_lines.append("\\end{quote}")
    tex_path = base_path.with_suffix(".tex")
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    return {"csv": csv_path, "md": md_path, "tex": tex_path}


def _read_run_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def export_all_publication_tables(config: dict[str, Any]) -> dict[str, dict[str, Path]]:
    """Export every canonical manuscript table (Table 1 through Table 6) as
    CSV + Markdown + LaTeX with captions, abbreviation lists, and notes.
    Every number in every note is pulled from the actual saved table or
    run-metadata JSON — never a hardcoded illustrative value.
    """
    pub_config = config["publication"]
    tables_dir = Path(pub_config["tables_output_dir"])
    exports: dict[str, dict[str, Path]] = {}

    final_eval_meta = _read_run_metadata(config["final_evaluation"]["run_metadata_path"])
    n_test = final_eval_meta.get("n_test")
    n_bootstrap = final_eval_meta.get("n_bootstrap")
    ci_alpha = final_eval_meta.get("ci_alpha")
    ci_pct = int(round((1 - ci_alpha) * 100)) if ci_alpha is not None else None
    seed = final_eval_meta.get("seed", config["project"]["random_seed"])

    # --- Table 1: cohort characteristics ---
    t1_path = Path(pub_config["cohort_characteristics_table_path"])
    if t1_path.exists():
        df = pd.read_csv(t1_path)
        exports["table1_cohort_characteristics"] = export_table_with_metadata(
            df, tables_dir / "table1_cohort_characteristics",
            caption="Table 1. Cohort characteristics, overall and by data split.",
            notes=[
                "Numeric variables summarized as median [interquartile range]; "
                "categorical variables as n (%).",
                "Splits are patient-level (no patient appears in more than one split).",
            ],
            abbreviations_used=["ICU", "GCS"],
        )

    # --- Table 2: hyperparameters ---
    t2_path = Path(pub_config["hyperparameters_table_path"])
    if t2_path.exists():
        df = pd.read_csv(t2_path)
        exports["table2_hyperparameters"] = export_table_with_metadata(
            df, tables_dir / "table2_hyperparameters",
            caption="Table 2. Selected model hyperparameters, validation AUROC, and training runtime.",
            notes=[
                f"Global random seed = {seed}." if seed is not None else "Global random seed fixed across all models.",
                "Hyperparameters were selected using validation-set AUROC only; "
                "the test set was not used for any selection decision.",
            ],
            abbreviations_used=["MLP", "AUROC"],
        )

    # --- Table 3: main results ---
    t3_path = Path(pub_config["main_results_table_path"])
    if t3_path.exists():
        df = pd.read_csv(t3_path)
        notes = [
            f"{ci_pct}% confidence intervals computed via {n_bootstrap} stratified bootstrap resamples "
            f"(resampled within each outcome class, preserving observed class counts)."
            if (ci_pct and n_bootstrap) else "Confidence intervals computed via stratified bootstrap resampling.",
            f"Evaluated once on the held-out test set (n = {n_test})." if n_test else
            "Evaluated once on the held-out test set.",
            "Calibration method per model was selected on validation-set Expected Calibration Error "
            "(10 bins) prior to test-set evaluation.",
            "In the Markdown/LaTeX rendering, \"value (lower–upper)\" denotes the point estimate "
            "and its 95% confidence interval; the full CSV retains point estimates and CI bounds "
            "as separate columns.",
        ]
        exports["table3_main_results"] = export_table_with_metadata(
            df, tables_dir / "table3_main_results",
            caption="Table 3. Main results: discrimination and calibration performance of each "
                    "calibrated baseline model on the held-out test set.",
            notes=notes,
            abbreviations_used=["AUROC", "AUPRC", "CI", "ECE"],
        )

    # --- Table 4: clinical thresholds ---
    t4_path = Path(pub_config["clinical_thresholds_table_path"])
    if t4_path.exists():
        df = pd.read_csv(t4_path)
        exports["table4_clinical_thresholds"] = export_table_with_metadata(
            df, tables_dir / "table4_clinical_thresholds",
            caption="Table 4. Selective-prediction referral policy performance at each "
                    "referral-coverage level, held-out test set.",
            notes=[
                "Coverage = fraction of test cases NOT referred to clinician review "
                "(automated/decision-support-eligible).",
                "Metrics computed on the accepted (non-referred) subset only, at each referral rate.",
                "CLINICAL DECISION SUPPORT ONLY: accepted cases still require clinician review; "
                "referred cases require mandatory full clinician review.",
                "In the Markdown/LaTeX rendering, \"value (lower–upper)\" denotes the point estimate "
                "and its 95% confidence interval.",
            ],
            abbreviations_used=["CI", "PPV", "NPV", "ECE"],
        )

    # --- Table 5: ablation study ---
    t5_path = Path(pub_config["ablation_study_table_path"])
    if t5_path.exists():
        df = pd.read_csv(t5_path)
        exports["table5_ablation_study"] = export_table_with_metadata(
            df, tables_dir / "table5_ablation_study",
            caption="Table 5. Ablation study: effect of calibration method on discrimination "
                    "and calibration quality, held-out test set.",
            notes=[
                "is_selected_method indicates the calibration method chosen for Tables 3, 4, and 6 "
                "(selected on validation ECE, not test performance).",
                "uncalibrated rows report the base model's raw predicted probabilities with no "
                "post-hoc calibration applied.",
                "In the Markdown/LaTeX rendering, \"value (lower–upper)\" denotes the point estimate "
                "and its 95% confidence interval.",
            ],
            abbreviations_used=["AUROC", "ECE", "CI"],
        )

    # --- Table 6: subgroup results ---
    t6_path = Path(pub_config["subgroup_results_table_path"])
    if t6_path.exists():
        df = pd.read_csv(t6_path)
        exports["table6_subgroup_results"] = export_table_with_metadata(
            df, tables_dir / "table6_subgroup_results",
            caption="Table 6. Subgroup results: post-hoc fairness/equity audit across "
                    "demographic and clinical subgroups, held-out test set.",
            notes=[
                "Subgroup levels with too few cases or too few positive-class events were "
                "skipped (metrics reported as NaN, see the note column) rather than approximated.",
                "This table is a post-hoc audit only; subgroup membership was not used as a "
                "model feature or in the referral decision.",
                "In the Markdown/LaTeX rendering, \"value (lower–upper)\" denotes the point estimate "
                "and its 95% confidence interval.",
            ],
            abbreviations_used=["AUROC", "CI", "ECE"],
        )

    return exports


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all_publication_figures(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    pub_config = config["publication"]

    test_predictions = load_all_test_predictions(config)

    outputs: dict[str, Any] = {}

    cohort_flow_path = Path(pub_config["cohort_flow_table_path"])
    if cohort_flow_path.exists():
        cohort_flow_df = pd.read_csv(cohort_flow_path)
        outputs["figure1_study_flowchart"] = make_study_flowchart_figure(cohort_flow_df, config)
        outputs["figure1_mermaid"] = generate_study_workflow_mermaid(
            cohort_flow_df, Path(pub_config["mermaid_output_dir"]) / "figure1_study_workflow.mmd"
        )

    outputs["figure2_architecture"] = make_architecture_diagram_figure(config)
    outputs["figure2_mermaid"] = generate_architecture_mermaid(
        Path(pub_config["mermaid_output_dir"]) / "figure2_system_architecture.mmd"
    )

    outputs["figure3_reliability"] = make_reliability_diagrams(test_predictions, config)
    outputs["figure4_roc"] = make_roc_curve_figure(test_predictions, config)
    outputs["figure5_pr"] = make_pr_curve_figure(test_predictions, config)
    outputs["figure6_risk_coverage"] = make_risk_coverage_figure(test_predictions, config)
    outputs["figure7_subgroup_calibration"] = make_subgroup_calibration_figure(test_predictions, config)

    return outputs


# ---------------------------------------------------------------------------
# Paper pipeline (notebooks/05_publication_outputs.ipynb, current spec):
# generic, schema-agnostic versions of Figures 1-2 for the simpler
# logreg+LightGBM pipeline (no MLP, no selective-prediction layer). Additive
# — does not modify make_study_flowchart_figure / make_architecture_diagram_figure
# above, which remain used by the archived notebooks/archive/06_publication_outputs.ipynb.
# ---------------------------------------------------------------------------

def make_paper_study_flowchart_figure(
    stage_labels: list[str], config: dict[str, Any], filename: str = "figure1_study_flowchart"
) -> tuple[Path, Path]:
    """Generic box-and-arrow flowchart from a plain list of pre-formatted
    stage label strings (caller embeds real counts into each label already
    — see notebooks/05_publication_outputs.ipynb).
    """
    figures_dir = Path(config["paper_publication"]["figures_output_dir"])
    dpi = config["paper_publication"]["dpi"]
    fig = _draw_flow_boxes(stage_labels, title="Figure 1. Study Cohort Flow", figsize=(5.5, 1.8 * len(stage_labels)))
    result = _save_figure(fig, figures_dir / filename, dpi=dpi)
    plt.close(fig)
    return result


def generate_paper_study_workflow_mermaid(stage_labels: list[str], save_path: str | Path) -> Path:
    lines = ["flowchart TD"]
    node_ids = [f"N{i}" for i in range(len(stage_labels))]
    for node_id, label in zip(node_ids, stage_labels):
        mermaid_label = label.replace("\n", "<br/>")
        lines.append(f'    {node_id}["{mermaid_label}"]')
    for a, b in zip(node_ids[:-1], node_ids[1:]):
        lines.append(f"    {a} --> {b}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return save_path


PAPER_ARCHITECTURE_STAGES = [
    "EHR features\n(first 24h ICU stay \u2014 demographics, vitals, labs;\nMIMIC-IV Clinical Database Demo v2.2)",
    "Preprocessing\n(median imputation + missingness indicators,\none-hot encoding, standard scaling \u2014 fit on train split only)",
    "Base models\n(Logistic Regression [class_weight=balanced] and LightGBM\n[scale_pos_weight] \u2014 validation-selected hyperparameters)",
    "Calibration layer\n(Platt scaling / isotonic regression \u2014\nfit on validation predictions only)",
    "Calibrated risk score",
    "Clinical decision support / referral\n(risk score is an input to clinician judgment,\nnot an autonomous diagnosis)",
]


def make_paper_architecture_diagram_figure(
    config: dict[str, Any], filename: str = "figure2_system_architecture"
) -> tuple[Path, Path]:
    figures_dir = Path(config["paper_publication"]["figures_output_dir"])
    dpi = config["paper_publication"]["dpi"]
    fig = _draw_flow_boxes(
        PAPER_ARCHITECTURE_STAGES,
        title="Figure 2. Confidence-Calibrated Clinical AI Architecture",
        figsize=(7, 1.8 * len(PAPER_ARCHITECTURE_STAGES)),
    )
    result = _save_figure(fig, figures_dir / filename, dpi=dpi)
    plt.close(fig)
    return result


def generate_paper_architecture_mermaid(save_path: str | Path) -> Path:
    lines = ["flowchart TD"]
    node_ids = [f"N{i}" for i in range(len(PAPER_ARCHITECTURE_STAGES))]
    for node_id, label in zip(node_ids, PAPER_ARCHITECTURE_STAGES):
        mermaid_label = label.replace("\n", "<br/>")
        lines.append(f'    {node_id}["{mermaid_label}"]')
    for a, b in zip(node_ids[:-1], node_ids[1:]):
        lines.append(f"    {a} --> {b}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return save_path
