from __future__ import annotations

import argparse

from hallucination_eval.metrics.debunking import debunking_report
from hallucination_eval.metrics.distortions import distortion_f1_report
from hallucination_eval.metrics.mcq_vcd import mcq_distortion_f1_report
from hallucination_eval.metrics.report import (
    binary_report,
    mcq_report,
    vcd_binary_report,
    vcd_black_report,
)
from hallucination_eval.metrics.semantic_entropy import se_report
from hallucination_eval.metrics.manual_eval import manual_eval_stats
from hallucination_eval.metrics.semantic_chair import chair_metrics, load_chair_evaluation
from hallucination_eval.metrics.bleu_cider import bleu_cider_report

_COLUMNS = ("precision", "recall", "f1", "accuracy", "yes_ratio", "roc_auc", "ap")


def _print_flat(rep: dict, prefix: str = "") -> None:
    for col in _COLUMNS:
        if col in rep:
            print(f"{prefix}{col:>12}: {rep[col]:.4f}")


def _dirs(specs: list[str]) -> dict[str, str]:
    """Parse ``--dir NAME=PATH`` specs into a dict."""
    out: dict[str, str] = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not path:
            raise SystemExit(f"--dir expects NAME=PATH, got: {spec!r}")
        out[name] = path
    return out


def _cmd_report(args: argparse.Namespace) -> None:
    _print_flat(binary_report(args.path, args.threshold))


def _cmd_mcq(args: argparse.Namespace) -> None:
    _print_flat(mcq_report(args.path, args.threshold))


def _cmd_vcd(args: argparse.Namespace) -> None:
    _print_flat(vcd_binary_report(args.original, args.distorted, args.alpha, args.beta, args.threshold))


def _cmd_vcd_black(args: argparse.Namespace) -> None:
    rep = vcd_black_report(args.original, args.black, args.alpha, args.beta, args.threshold)
    for label in ("baseline", "black_image"):
        print(f"--- {label} ---")
        _print_flat(rep[label])


def _cmd_distortion(args: argparse.Namespace) -> None:
    rep = distortion_f1_report(args.original, _dirs(args.dir), args.alpha, args.beta, args.threshold)
    for name, curve in rep.items():
        points = ", ".join(f"{lvl:.2f}={f1:.4f}" for lvl, f1 in curve)
        print(f"{name:>16}: {points}")


def _cmd_mcq_vcd(args: argparse.Namespace) -> None:
    rep = mcq_distortion_f1_report(_dirs(args.dir), args.alpha, args.beta)
    for name, curve in rep.items():
        points = ", ".join(f"{lvl:.2f}={f1:.4f}" for lvl, f1 in curve)
        print(f"{name:>16}: {points}")


def _cmd_debunking(args: argparse.Namespace) -> None:
    rep = debunking_report(args.results_base)
    for strat, m in rep.items():
        print(f"{strat:>16}: F1={m['f1']:.4f}  yes_ratio={m['yes_ratio']:.4f}  n={m['n']}")


def _cmd_se(args: argparse.Namespace) -> None:
    rep = se_report(args.path)
    for key, val in rep.items():
        if isinstance(val, float):
            print(f"{key:>18}: {val:.4f}")
        else:
            print(f"{key:>18}: {val}")


def _cmd_manual_eval(args: argparse.Namespace) -> None:
    rep = manual_eval_stats(args.path)
    print(f"       total_objects: {rep['total_objects']}")
    print(f"  not_visible_objects: {rep['not_visible_objects']}")
    print(f"object_hallucination_rate: {rep['object_hallucination_rate']:.4f}")
    print(f"         total_images: {rep['total_images']}")
    print(f"images_with_not_visible: {rep['images_with_not_visible']}")
    print(f" image_hallucination_rate: {rep['image_hallucination_rate']:.4f}")


def _cmd_chair(args: argparse.Namespace) -> None:
    rep = chair_metrics(load_chair_evaluation(args.path))
    for label in ("original", "revised"):
        d = rep[label]
        print(
            f"{label:>8}: CHAIR_i={d['chair_i']:.4f} ({d['hallucinated_objects']}/{d['total_objects']})"
            f"  CHAIR_s={d['chair_s']:.4f} ({d['sentences_with_hallucinations']}/{d['total_sentences']})"
        )


def _cmd_bleu_cider(args: argparse.Namespace) -> None:
    rep = bleu_cider_report(args.dataset)
    print(f"{rep['name']}: n={rep['num_samples']}")
    for field in ("original", "revised"):
        bleu = " ".join(f"{x:.4f}" for x in rep["bleu"][field])
        print(f"  {field:>8}: BLEU=[{bleu}]  CIDEr={rep['cider'][field]:.4f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hallucination-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    def _threshold(p: argparse.ArgumentParser) -> None:
        p.add_argument("--threshold", type=float, default=0.5)

    def _alpha_beta(p: argparse.ArgumentParser) -> None:
        p.add_argument("--alpha", type=float, default=1.0)
        p.add_argument("--beta", type=float, default=0.1)

    r = sub.add_parser("report", help="binary detection report (A/B result JSON)")
    r.add_argument("path", help="path to an A/B logit result JSON")
    _threshold(r)
    r.set_defaults(func=_cmd_report)

    m = sub.add_parser("mcq", help="MCQ detection report")
    m.add_argument("path", help="path to an MCQ result JSON")
    _threshold(m)
    m.set_defaults(func=_cmd_mcq)

    v = sub.add_parser("vcd", help="binary VCD+APC report (original vs distorted logits)")
    v.add_argument("original")
    v.add_argument("distorted")
    _alpha_beta(v)
    _threshold(v)
    v.set_defaults(func=_cmd_vcd)

    vb = sub.add_parser("vcd-black", help="tab:vcd_performance (baseline + black image VCD)")
    vb.add_argument("original")
    vb.add_argument("black")
    _alpha_beta(vb)
    _threshold(vb)
    vb.set_defaults(func=_cmd_vcd_black)

    d = sub.add_parser("distortion", help="binary VCD distortion-F1 curves")
    d.add_argument("original")
    d.add_argument("--dir", action="append", required=True, metavar="NAME=PATH", help="repeatable")
    _alpha_beta(d)
    _threshold(d)
    d.set_defaults(func=_cmd_distortion)

    mv = sub.add_parser("mcq-vcd", help="MCQ+VCD distortion Rej-F1 curves")
    mv.add_argument("--dir", action="append", required=True, metavar="NAME=PATH", help="repeatable")
    mv.add_argument("--alpha", type=float, default=1.0)
    mv.add_argument("--beta", type=float, default=0.2)
    mv.set_defaults(func=_cmd_mcq_vcd)

    db = sub.add_parser("debunking", help="six decoding strategies (F1 + yes-ratio)")
    db.add_argument("results_base", help="{model}/{dataset} base, e.g. .../llava-1.5-7b/pope")
    db.set_defaults(func=_cmd_debunking)

    s = sub.add_parser("se", help="semantic-entropy uncertainty report")
    s.add_argument("path", help="qa_semantic_entropy_*.jsonl file")
    s.set_defaults(func=_cmd_se)

    me = sub.add_parser("manual-eval", help="manual-evaluation stats (§5.2.3)")
    me.add_argument("path", help="manual_evaluations.jsonl file")
    me.set_defaults(func=_cmd_manual_eval)

    ch = sub.add_parser("chair", help="Semantic-CHAIR (§5.4)")
    ch.add_argument("path", help="chair_evaluation_*.jsonl file")
    ch.set_defaults(func=_cmd_chair)

    bc = sub.add_parser("bleu-cider", help="BLEU/CIDEr caption metrics (§5.4)")
    bc.add_argument("dataset", choices=["GAIA", "Git-10M", "DDFAV"])
    bc.set_defaults(func=_cmd_bleu_cider)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
