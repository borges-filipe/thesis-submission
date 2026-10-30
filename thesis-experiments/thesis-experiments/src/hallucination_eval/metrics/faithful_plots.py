from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

sns.set_theme(style="whitegrid")

CM = 1 / 2.54
HALF = 15.25 / 2


def _save(fig, path, dpi=600):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(path + ".png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _optimal_f1_threshold(y_true, y_score):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1 = np.divide(
        2 * precisions * recalls, precisions + recalls,
        out=np.zeros_like(precisions), where=(precisions + recalls) != 0,
    )
    idx = int(np.argmax(f1[:-1]))
    return float(thresholds[idx])


# --------------------------------------------------------------------------
# binary / MCQ dashboard (binary_hall_detect_performance.py / mcq_hal_*)
# --------------------------------------------------------------------------

def binary_dashboard(y_true, y_score_hal, out_dir, name) -> None:
    """Histogram + ROC + separate legend files """
    os.makedirs(f"{out_dir}/histograms", exist_ok=True)
    os.makedirs(f"{out_dir}/roc_curves", exist_ok=True)

    y_true = np.asarray(y_true)
    y_score_hal = np.asarray(y_score_hal)
    conf = 1.0 - y_score_hal  # confidence_visible (histogram x-axis)

    df = pd.concat([
        pd.DataFrame({"Score": conf[y_true == 0], "Class": "Non-Hal"}),
        pd.DataFrame({"Score": conf[y_true == 1], "Class": "Hal"}),
    ])

    opt_thresh_hal = _optimal_f1_threshold(y_true, y_score_hal)
    opt_thresh_raw = 1.0 - opt_thresh_hal

    fig_hist, ax_hist = plt.subplots(1, 1, figsize=(HALF * CM, 1.6), dpi=50)
    fig_roc, ax_roc = plt.subplots(1, 1, figsize=(HALF * CM, 1.6), dpi=50)

    sns.histplot(
        data=df, x="Score", hue="Class", element="step", stat="count",
        common_norm=False, bins=30, alpha=0.4, ax=ax_hist,
        kde=True, kde_kws={"bw_adjust": 0.4},
    )

    legend_hist = ax_hist.get_legend()
    if legend_hist is not None:
        legend_hist.set_title("Class")
        legend_hist.get_title().set_fontsize(6)
        for text in legend_hist.get_texts():
            text.set_fontsize(6)
        class_handles = legend_hist.legend_handles
        class_labels = [t.get_text() for t in legend_hist.get_texts()]
        legend_hist.remove()
    else:
        class_handles, class_labels = [], []

    threshold_line = ax_hist.axvline(x=0.5, color="gray", linestyle="--", label="Threshold (0.5)")
    optimal_f1_line = ax_hist.axvline(x=opt_thresh_raw, color="red", alpha=0.4, linestyle="--", label="Optimal F1")
    ax_hist.set_xlabel("confidence visible (0 to 1)", fontsize=8)
    ax_hist.set_ylabel("n Samples", fontsize=8)
    ax_hist.set_xlim(0, 1)
    ax_hist.tick_params(axis="both", labelsize=6)

    fpr, tpr, thresholds_roc = roc_curve(y_true, y_score_hal)
    roc_auc = float(roc_auc_score(y_true, y_score_hal))
    roc_optimal_idx = int(np.argmin(np.abs(thresholds_roc - opt_thresh_hal)))
    model_line = ax_roc.plot(fpr, tpr, label="Model")[0]
    opt_marker = ax_roc.plot(
        fpr[roc_optimal_idx], tpr[roc_optimal_idx], marker="o", color="red",
        label="Optimal Threshold (Max F1)",
    )[0]
    random_line = ax_roc.plot(
        [0, 1], [0, 1], color="gray", linestyle="--", label="Random Guessing (0.500)",
    )[0]
    ax_roc.set_xlabel("False Positive Rate", fontsize=8)
    ax_roc.set_ylabel("Recall (TP Rate)", fontsize=8)
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.tick_params(axis="both", labelsize=6)

    legend_fig_hist = plt.figure(figsize=(HALF * CM, 0.7), dpi=50)
    legend_fig_hist.subplots_adjust(left=0, right=1, top=1, bottom=0)
    hist_legend = legend_fig_hist.legend(
        class_handles + [threshold_line, optimal_f1_line],
        class_labels + ["Threshold (0.5)", "Optimal F1"],
        loc="center", frameon=True, fontsize=8, ncol=1,
    )
    legend_fig_hist.add_artist(hist_legend)
    legend_fig_hist.savefig(f"{out_dir}/histograms/{name}_legend.png", dpi=600, transparent=True)
    legend_fig_hist.savefig(f"{out_dir}/histograms/{name}_legend.pdf", dpi=600, transparent=True)
    plt.close(legend_fig_hist)

    legend_fig_roc = plt.figure(figsize=(HALF * CM, 0.7), dpi=50)
    legend_fig_roc.subplots_adjust(left=0, right=1, top=1, bottom=0)
    roc_legend = legend_fig_roc.legend(
        [model_line, opt_marker, random_line],
        ["Model", "Optimal Threshold (Max F1)", "Random Guessing (0.500)"],
        loc="center", frameon=True, fontsize=8, ncol=1,
    )
    legend_fig_roc.add_artist(roc_legend)
    legend_fig_roc.savefig(f"{out_dir}/roc_curves/{name}_legend.png", dpi=600, transparent=True)
    legend_fig_roc.savefig(f"{out_dir}/roc_curves/{name}_legend.pdf", dpi=600, transparent=True)
    plt.close(legend_fig_roc)

    fig_hist.tight_layout()
    fig_hist.savefig(f"{out_dir}/histograms/{name}_histogram.png", dpi=600, bbox_inches="tight")
    fig_hist.savefig(f"{out_dir}/histograms/{name}_histogram.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig_hist)

    fig_roc.tight_layout()
    fig_roc.savefig(f"{out_dir}/roc_curves/{name}_roc.png", dpi=600, bbox_inches="tight")
    fig_roc.savefig(f"{out_dir}/roc_curves/{name}_roc.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig_roc)


# --------------------------------------------------------------------------
# Semantic Entropy dashboard (se_data_separation.py)
# --------------------------------------------------------------------------

def se_dashboard(y_correct, scores, out_dir, name) -> None:
    """Entropy histogram + rejection-accuracy curve + separate legends."""
    os.makedirs(f"{out_dir}/histograms", exist_ok=True)
    os.makedirs(f"{out_dir}/rac_curves", exist_ok=True)

    y_correct = np.asarray(y_correct, dtype=bool)
    scores = np.asarray(scores)

    y_incorrect = (~y_correct).astype(int)
    df = pd.concat([
        pd.DataFrame({"Score": scores[y_correct], "Class": "Correct"}),
        pd.DataFrame({"Score": scores[~y_correct], "Class": "Incorrect"}),
    ])

    optimal_thresh = _optimal_f1_threshold(y_incorrect, scores)
    median_thresh = float(np.median(scores))

    fig_hist, ax_hist = plt.subplots(1, 1, figsize=(HALF * CM, 1.6), dpi=50)
    fig_rac, ax_rac = plt.subplots(1, 1, figsize=(HALF * CM, 1.6), dpi=50)

    sns.histplot(
        data=df, x="Score", hue="Class", element="step", stat="count",
        common_norm=False, bins=30, alpha=0.4, ax=ax_hist,
        kde_kws={"bw_adjust": 0.4}, legend=False,
    )

    ax_hist.axvline(x=median_thresh, color="gray", linestyle="--")
    ax_hist.axvline(x=optimal_thresh, color="red", alpha=0.4, linestyle="--")
    ax_hist.set_xlabel("Semantic Entropy Score", fontsize=8)
    ax_hist.set_ylabel("Number of Samples", fontsize=8)
    ax_hist.tick_params(axis="both", labelsize=6)

    patch_correct = mpatches.Patch(color="C0", label="Correct", alpha=0.4)
    patch_incorrect = mpatches.Patch(color="C1", label="Incorrect", alpha=0.4)
    line_median = mlines.Line2D([], [], color="gray", linestyle="--", label="Median")
    line_opt = mlines.Line2D([], [], color="red", alpha=0.4, linestyle="--", label="Optimal Rej. F1")

    order = np.argsort(scores)
    sorted_correct = y_correct[order].astype(float)
    k = np.arange(1, len(sorted_correct) + 1)
    accuracies = np.cumsum(sorted_correct) / k
    rejections = 1 - k / len(sorted_correct)

    kept_mask = scores < optimal_thresh
    num_kept = int(np.sum(kept_mask))
    opt_rejection = 1 - num_kept / len(scores)
    opt_accuracy = float(np.mean(y_correct[kept_mask])) if num_kept > 0 else 0.0

    model_line = ax_rac.plot(rejections, accuracies, label="Model")[0]
    opt_marker = ax_rac.plot(
        opt_rejection, opt_accuracy, marker="o", color="red",
        label="Optimal Threshold (Max Rej. F1)",
    )[0]
    baseline_accuracy = float(np.mean(y_correct))
    baseline_line = ax_rac.axhline(y=baseline_accuracy, color="gray", linestyle="--", label="Baseline Accuracy")

    ax_rac.set_xlabel("Rejection Rate", fontsize=8)
    ax_rac.set_ylabel("Kept Samples Acc.", fontsize=8)
    ax_rac.set_xlim([-0.02, 1.02])
    ax_rac.set_ylim([max(0.0, baseline_accuracy - 0.1), 1.02])
    ax_rac.tick_params(axis="both", labelsize=6)

    legend_fig_hist = plt.figure(figsize=(HALF * CM, 0.7), dpi=50)
    legend_fig_hist.subplots_adjust(left=0, right=1, top=1, bottom=0)
    hist_legend = legend_fig_hist.legend(
        handles=[patch_correct, patch_incorrect, line_median, line_opt],
        loc="center", frameon=True, fontsize=8, ncol=1,
    )
    legend_fig_hist.add_artist(hist_legend)
    legend_fig_hist.savefig(f"{out_dir}/histograms/{name}_cap_legend.png", dpi=600, transparent=True)
    legend_fig_hist.savefig(f"{out_dir}/histograms/{name}_cap_legend.pdf", dpi=600, transparent=True)
    plt.close(legend_fig_hist)

    handles_rac, labels_rac = ax_rac.get_legend_handles_labels()
    legend_fig_rac = plt.figure(figsize=(HALF * CM, 0.7), dpi=50)
    legend_fig_rac.subplots_adjust(left=0, right=1, top=1, bottom=0)
    rac_legend = legend_fig_rac.legend(handles_rac, labels_rac, loc="center", frameon=True, fontsize=8, ncol=1)
    legend_fig_rac.add_artist(rac_legend)
    legend_fig_rac.savefig(f"{out_dir}/rac_curves/{name}_cap_legend.png", dpi=600, transparent=True)
    legend_fig_rac.savefig(f"{out_dir}/rac_curves/{name}_cap_legend.pdf", dpi=600, transparent=True)
    plt.close(legend_fig_rac)

    fig_hist.tight_layout()
    fig_hist.savefig(f"{out_dir}/histograms/{name}_cap_histogram.png", dpi=600, bbox_inches="tight")
    fig_hist.savefig(f"{out_dir}/histograms/{name}_cap_histogram.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig_hist)

    fig_rac.tight_layout()
    fig_rac.savefig(f"{out_dir}/rac_curves/{name}_cap_rac.png", dpi=600, bbox_inches="tight")
    fig_rac.savefig(f"{out_dir}/rac_curves/{name}_cap_rac.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig_rac)


# --------------------------------------------------------------------------
# Distortion F1 curves (vcd_distortions_auroc_aurac.py)
# --------------------------------------------------------------------------

def _distortion_colors(dists):
    all_unique = sorted(dists)
    color_mae = [d for d in all_unique if d.startswith("color_mae")]
    others = [d for d in all_unique if not d.startswith("color_mae") and d != "gaussian_noise"]
    color_map = {}
    if len(color_mae) == 1:
        color_map[color_mae[0]] = "0.5"
    elif len(color_mae) > 1:
        for i, d in enumerate(color_mae):
            gray = 0.5 + 0.3 * (i / (len(color_mae) - 1))
            color_map[d] = str(round(gray, 2))
    palette = sns.color_palette("tab10", n_colors=len(others))
    for i, d in enumerate(others):
        color_map[d] = palette[i % len(palette)]
    return color_map


def distortion_f1_figure(curves: dict[str, list[tuple[float, float]]], baseline_f1: float,
                         out_dir: str, name: str, save_legend: bool = False) -> None:
    """F1 (hallucination) vs relative intensity lineplot"""
    os.makedirs(out_dir, exist_ok=True)

    color_map = _distortion_colors(list(curves.keys()))

    fig, ax = plt.subplots(1, 1, figsize=(HALF * CM, 1.6), dpi=50)
    data_plotted = False

    for dist in sorted(curves, reverse=True):
        curve = curves[dist]
        if not curve:
            continue
        levels = [l for l, _ in curve]
        f1s = [f for _, f in curve]
        min_l, max_l = min(levels), max(levels)
        rel = [(l - min_l) / (max_l - min_l) if max_l > min_l else 0.5 for l in levels]
        sns.lineplot(x=rel, y=f1s, marker="o", linewidth=1.5, markersize=4,
                     label=dist, color=color_map.get(dist, "black"), ax=ax)
        data_plotted = True

    if data_plotted:
        ax.axhline(y=baseline_f1, color="red", linestyle="--", linewidth=1.5, zorder=10, label="Baseline")

    ax.set_xlabel("Relative Intensity", fontsize=8)
    ax.set_ylabel("F1 Score", fontsize=8)
    ax.tick_params(axis="both", labelsize=6)
    if ax.get_legend():
        ax.get_legend().remove()

    if save_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            legend_fig = plt.figure(figsize=(15.25 * CM, 0.7), dpi=50)
            legend_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            legend_fig.legend(by_label.values(), by_label.keys(), loc="center",
                              frameon=True, fontsize=6, ncol=len(by_label) // 2 or 1)
            legend_fig.savefig(f"{out_dir}/global_distortion_legend.pdf", dpi=600, bbox_inches="tight", transparent=True)
            legend_fig.savefig(f"{out_dir}/global_distortion_legend.png", dpi=600, bbox_inches="tight", transparent=True)
            plt.close(legend_fig)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/{name}_f1_hallucination.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out_dir}/{name}_f1_hallucination.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def mcq_distortion_f1_figure(curves: dict[str, list[tuple[float, float]]],
                             out_dir: str, name: str, save_legend: bool = False) -> None:
    """MCQ+VCD Rej F1 vs relative intensity"""
    os.makedirs(out_dir, exist_ok=True)

    color_map = _distortion_colors(list(curves.keys()))

    fig, ax = plt.subplots(1, 1, figsize=(HALF * CM, 1.6), dpi=50)

    for dist in sorted(curves, reverse=True):
        curve = curves[dist]
        if not curve:
            continue
        levels = [l for l, _ in curve]
        f1s = [f for _, f in curve]
        min_l, max_l = min(levels), max(levels)
        rel = [(l - min_l) / (max_l - min_l) if max_l > min_l else 0.5 for l in levels]
        sns.lineplot(x=rel, y=f1s, marker="o", linewidth=1.5, markersize=4,
                     label=dist, color=color_map.get(dist, "black"), ax=ax)

    ax.set_xlabel("Relative Intensity", fontsize=8)
    ax.set_ylabel("F1 Score (0.5)", fontsize=8)
    ax.tick_params(axis="both", labelsize=6)
    if ax.get_legend():
        ax.get_legend().remove()

    if save_legend:
        legend_dir = f"{out_dir}/legends"
        os.makedirs(legend_dir, exist_ok=True)
        handles = [
            mlines.Line2D([], [], color=color_map.get(d, "black"), marker="o",
                          linewidth=1.5, markersize=4)
            for d in sorted(curves, reverse=True) if curves[d]
        ]
        labels = [d for d in sorted(curves, reverse=True) if curves[d]]
        handles.append(mlines.Line2D([], [], color="red", linestyle="--", linewidth=1.5))
        labels.append("Baseline")
        legend_fig = plt.figure(figsize=(HALF * CM, 0.7), dpi=50)
        legend_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        legend_fig.legend(handles, labels, loc="center", frameon=True, fontsize=8, ncol=1)
        legend_fig.savefig(f"{legend_dir}/{name}_legend.pdf", dpi=600, bbox_inches="tight", transparent=True)
        legend_fig.savefig(f"{legend_dir}/{name}_legend.png", dpi=600, bbox_inches="tight", transparent=True)
        plt.close(legend_fig)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/model_summary_{name}_rej_f1.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out_dir}/model_summary_{name}_rej_f1.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Debunking (vcd_debunking_bar_plots_diff.py)
# --------------------------------------------------------------------------

def debunking_figure(report: dict[str, dict], out_dir: str, dataset: str) -> None:
    """Two-panel F1-improvement + yes-ratio bar chart"""
    os.makedirs(out_dir, exist_ok=True)

    order = ["regular", "vcd", "apc", "greedy", "binary_decoding", "binary_vcd"]
    labels = ["Regular\nDecoding", "VCD", "APC", "Greedy\nDecoding", "Binary\nDecoding", "Binary\nVCD"]
    baseline_f1 = report["regular"]["f1"]
    f1_imp = [report[k]["f1"] - baseline_f1 for k in order]
    yes_ratios = [report[k]["yes_ratio"] * 100 for k in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.25 * CM, 3), dpi=50)

    sns.barplot(x=labels, y=f1_imp, ax=ax1, hue=labels, palette="Set2", legend=False)
    ax1.axhline(0, color="black", linestyle="--")
    ax1.set_title("F1-Score Improvement (vs. Regular Decoding)", fontsize=8)
    ax1.set_ylabel("Delta F1-Score", fontsize=8)
    ax1.tick_params(axis="x", rotation=45, labelsize=6)
    ax1.tick_params(axis="y", labelsize=6)
    min_imp, max_imp = min(f1_imp), max(f1_imp)
    padding = (max_imp - min_imp) * 0.15 if max_imp != min_imp else 0.05
    ax1.set_ylim(min_imp - padding, max_imp + padding)
    for i, v in enumerate(f1_imp):
        abs_f1 = baseline_f1 + v
        if i == 0:
            ax1.text(i, v + padding * 0.1, f" Baseline\n  F1: {abs_f1:.3f}",
                     ha="center", va="bottom", fontsize=5.5, fontweight="bold", color="gray")
        else:
            sign = "+" if v > 0 else ""
            va = "bottom" if v >= 0 else "top"
            off = padding * 0.1 if v >= 0 else -padding * 0.1
            ax1.text(i, v + off, f"{sign}{v:.3f}", ha="center", va=va, fontsize=6)

    sns.barplot(x=labels, y=yes_ratios, ax=ax2, hue=labels, palette="Set2", legend=False)
    ax2.set_title("Overall Yes Ratio (%)", fontsize=8)
    ax2.set_ylabel("Yes Ratio (%)", fontsize=8)
    max_ratio = max(yes_ratios) if yes_ratios else 100
    ax2.set_ylim(0, max(100, max_ratio * 1.15))
    ax2.tick_params(axis="x", rotation=45, labelsize=6)
    ax2.tick_params(axis="y", labelsize=6)
    for i, v in enumerate(yes_ratios):
        ax2.text(i, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=6)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/f1_and_yes_ratios_comparison_{dataset}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/f1_and_yes_ratios_comparison_{dataset}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
