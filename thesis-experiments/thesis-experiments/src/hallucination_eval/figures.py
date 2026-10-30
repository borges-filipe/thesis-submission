from __future__ import annotations

import argparse
import glob
import os

from hallucination_eval.data.ab_logits import load_binary_results
from hallucination_eval.data.mcq_logits import load_mcq_results
from hallucination_eval.metrics.debunking import debunking_report
from hallucination_eval.metrics.distortions import distortion_f1_report
from hallucination_eval.metrics.mcq_vcd import mcq_distortion_f1_report
from hallucination_eval.metrics import faithful_plots
from hallucination_eval.metrics.report import binary_report
from hallucination_eval.metrics.semantic_entropy import load_se_pairs

BASE = os.environ.get("HALLUCINATION_RESULTS_ROOT", "/mnt/storagecube/filipe/out/results")

# model -> LID original logits path (relative to results root)
MODELS = {
    "llava-1.5-7b": "lid/binary/logits/AB_LID_balanced.json",
    "rs-llava": "lid/binary/logits/AB_LID_balanced.json",
    "qwen3-8b": "lid/AB_LID_balanced.json",
    "qwen3.5-9b": "lid/AB_LID_balanced.json",
    "geochat": "lid/binary/logits/AB_LID_balanced.json",
}

DISTORTIONS = {
    "speckle": "speckle_noise",
    "diffusion": "diffusion_noise",
    "color_mae_red": "color_mae_masking/red",
    "color_mae_blue": "color_mae_masking/blue",
    "color_mae_green": "color_mae_masking/green",
    "color_mae_purple": "color_mae_masking/purple",
}

# MCQ+VCD distortion directories (Fig 5.9) — gaussian_noise is excluded.
MCQ_DISTORTIONS = [
    "color_mae_blue",
    "color_mae_green",
    "color_mae_purple",
    "color_mae_red",
    "diffusion",
    "speckle",
]


def gen_baseline(out: str) -> None:
    for ds in ["rspope", "lid"]:
        for model in MODELS:
            if ds == "rspope":
                path = f"{BASE}/{model}/rspope/AB_rspope.json"
            else:
                path = f"{BASE}/{model}/{MODELS[model]}"
            y_true, y_score_hal = load_binary_results(path)
            faithful_plots.binary_dashboard(
                y_true, y_score_hal, f"{out}/baseline_comparison_individual_plots/{ds}", model
            )


def gen_mcq(out: str) -> None:
    for model in MODELS:
        matches = glob.glob(f"{BASE}/{model}/rspope/mcq/logits/MCQ_logits_*_RSPOPE_MCQ.json")
        if not matches:
            continue
        y_true, y_score_hal = load_mcq_results(matches[0])
        faithful_plots.binary_dashboard(y_true, y_score_hal, f"{out}/mcq_individial_plots/rspope", model)


def gen_distortion(out: str) -> None:
    for ds, sub in [("rspope", "vcd_individual_plots"), ("lid", "vcd_lid_individual_plots")]:
        legend_saved = False
        for model, lid_og in MODELS.items():
            if ds == "rspope":
                original = f"{BASE}/{model}/rspope/AB_rspope.json"
                root = f"{BASE}/{model}/rspope"
            else:
                original = f"{BASE}/{model}/{lid_og}"
                root = f"{BASE}/{model}/lid/all_noise"
            dirs = {name: f"{root}/{rel}" for name, rel in DISTORTIONS.items()}
            curves = distortion_f1_report(original, dirs)
            baseline_f1 = binary_report(original)["f1"]
            out_dir = f"{out}/{sub}/f1_score_hallucination"
            faithful_plots.distortion_f1_figure(curves, baseline_f1, out_dir, model, save_legend=not legend_saved)
            legend_saved = True


def gen_mcq_vcd(out: str) -> None:
    for model in MODELS:
        root = f"{BASE}/{model}/rspope/mcq/logits"
        dirs = {d: f"{root}/{d}" for d in MCQ_DISTORTIONS}
        if not os.path.isdir(root):
            continue
        curves = mcq_distortion_f1_report(dirs)
        faithful_plots.mcq_distortion_f1_figure(
            curves, f"{out}/mcq_vcd_individual_plots", model, save_legend=True
        )


def gen_debunking(out: str) -> None:
    for ds in ["pope", "rspope"]:
        rep = debunking_report(f"{BASE}/llava-1.5-7b/{ds}")
        faithful_plots.debunking_figure(rep, f"{out}/vcd_debunking/llava", ds)


def _se_path(model: str, tag: str) -> str:
    if tag == "vqa":
        files = glob.glob(f"{BASE}/{model}/git_vqa_semantic_entropy_ddfav/qa_semantic_entropy_DDFAV_*.jsonl")
    else:
        files = glob.glob(f"{BASE}/{model}/git_vqa_semantic_entropy/qa_semantic_entropy_*.jsonl")
    return sorted(files)[-1] if files else ""


def gen_se(out: str) -> None:
    for tag in ["vqa", "captions"]:
        for model in MODELS:
            path = _se_path(model, tag)
            if not path:
                continue
            y_correct, scores = load_se_pairs(path)
            faithful_plots.se_dashboard(y_correct, scores, f"{out}/se_metrics/{model}/{tag}", model)


def generate_all(out: str) -> None:
    gen_baseline(out)
    gen_mcq(out)
    gen_distortion(out)
    gen_mcq_vcd(out)
    gen_debunking(out)
    gen_se(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all thesis figures.")
    parser.add_argument("--out", default="outputs/figures")
    args = parser.parse_args()
    generate_all(args.out)
    print(f"figures written to {args.out}")
