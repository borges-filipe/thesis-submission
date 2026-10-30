from __future__ import annotations

import glob
import re
from pathlib import Path

from hallucination_eval.metrics.report import vcd_binary_report


def _levels_in(directory: Path) -> list[tuple[float, Path]]:
    out: list[tuple[float, Path]] = []
    for f in glob.glob(str(directory / "level_*.json")):
        p = Path(f)
        m = re.search(r"level_(\d+(?:\.\d+)?)", p.name)
        if m:
            out.append((float(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def distortion_f1_report(
    original_path: str | Path,
    distorted_dirs: dict[str, str | Path],
    alpha: float = 1.0,
    beta: float = 0.1,
    threshold: float = 0.5,
) -> dict[str, list[tuple[float, float]]]:

    report: dict[str, list[tuple[float, float]]] = {}
    for name, directory in distorted_dirs.items():
        curve: list[tuple[float, float]] = []
        for level, path in _levels_in(Path(directory)):
            metrics = vcd_binary_report(original_path, path, alpha, beta, threshold)
            curve.append((level, metrics["f1"]))
        report[name] = curve
    return report
