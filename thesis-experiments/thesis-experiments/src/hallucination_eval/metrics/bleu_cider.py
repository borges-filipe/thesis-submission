from __future__ import annotations

import json
import os
from glob import glob

# Data paths (matching the original script).
GAIA_GT_FILE = "/mnt/storagecube/filipe/GAIA/gt_extracted_objects/gaia-test-object-extraction-20260810_131928.jsonl"
GAIA_GEN_FILE = "/mnt/storagecube/filipe/GAIA/gen_extracted_objects/gaia-gen-object-extraction-20260810_153837.jsonl"
GAIA_REVISED_FILE = "/mnt/storagecube/filipe/GAIA/revised_extracted_objects/gaia-revised-object-extraction-20260810_162611.jsonl"

GIT10M_TEST_FILE = "/mnt/storagecube/filipe/out/lure_finetune_data/test_split/extracted_objects/revised_captions-object-extraction-20260722_091425.jsonl"

DDFAV_BASE_DIR = "/mnt/storagecube/filipe/rspope/DDFAV_Instruction_Set"
DDFAV_TEST_DIR = os.path.join(DDFAV_BASE_DIR, "revised_captions", "extracted_objects")
DDFAV_SUBSETS = ["ai-tod", "dior", "dota", "fair1m", "visdrone-2019"]


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_jsonl_keyed(path, key_field):
    data = {}
    for item in _read_jsonl(path):
        key = item.get(key_field)
        if key is not None:
            data[key] = item
    return data


def _tokenize(text):
    return str(text).lower().split()


def load_gaia():
    gt = _load_jsonl_keyed(GAIA_GT_FILE, "key")
    gen = _load_jsonl_keyed(GAIA_GEN_FILE, "key")
    rev = _load_jsonl_keyed(GAIA_REVISED_FILE, "key")

    samples = []
    for key in sorted(set(gt) & set(gen) & set(rev)):
        samples.append({
            "id": str(key),
            "refs": [r for r in gt[key].get("captions", []) if str(r).strip()],
            "original": gen[key].get("generated_caption", ""),
            "revised": rev[key].get("revised_caption", ""),
        })
    return samples


def load_git10m():
    samples = []
    for item in _read_jsonl(GIT10M_TEST_FILE):
        caption = item.get("caption", "")
        samples.append({
            "id": str(item.get("img_name", "")),
            "refs": [caption] if str(caption).strip() else [],
            "original": item.get("generated_caption", ""),
            "revised": item.get("revised_caption", ""),
        })
    return samples


def load_ddfav():
    refs_by_image = {}
    for subset in DDFAV_SUBSETS:
        path = os.path.join(DDFAV_BASE_DIR, f"{subset}.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            conversations = item.get("conversations", []) or []
            first_gpt = next(
                (c.get("value", "") for c in conversations if c.get("from") == "gpt"),
                "",
            )
            refs_by_image[item.get("image")] = first_gpt

    samples = []
    for path in sorted(glob(os.path.join(DDFAV_TEST_DIR, "*.jsonl"))):
        for item in _read_jsonl(path):
            image = item.get("image", "")
            ref = refs_by_image.get(image, "")
            samples.append({
                "id": str(image),
                "refs": [ref] if str(ref).strip() else [],
                "original": item.get("generated_caption", ""),
                "revised": item.get("revised_caption", ""),
            })
    return samples


def compute_bleu(samples):
    """BLEU"""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    smooth = SmoothingFunction().method1
    accum = {"original": [0.0] * 4, "revised": [0.0] * 4}
    count = 0

    for sample in samples:
        ref_tokens = [_tokenize(r) for r in sample["refs"] if str(r).strip()]
        if not ref_tokens:
            continue
        count += 1
        for field in ("original", "revised"):
            candidate = _tokenize(sample[field])
            if not candidate:
                continue
            for n in range(1, 5):
                weights = [1.0 / n] * n + [0.0] * (4 - n)
                try:
                    bleu = sentence_bleu(ref_tokens, candidate, weights=weights, smoothing_function=smooth)
                except Exception:
                    bleu = 0.0
                accum[field][n - 1] += bleu

    if count == 0:
        return {"original": [0.0] * 4, "revised": [0.0] * 4}
    return {field: [v / count for v in accum[field]] for field in ("original", "revised")}


def compute_cider(samples):
    """CIDEr"""
    from pycocoevalcap.cider.cider import Cider

    gts, original_res, revised_res = {}, {}, {}
    for sample in samples:
        if not sample["refs"]:
            continue
        key = sample["id"]
        gts[key] = [str(r) for r in sample["refs"]]
        original_res[key] = [str(sample["original"])]
        revised_res[key] = [str(sample["revised"])]

    if not gts:
        return {"original": 0.0, "revised": 0.0}

    scorer = Cider()
    original_score, _ = scorer.compute_score(gts, original_res)
    revised_score, _ = scorer.compute_score(gts, revised_res)
    return {"original": original_score, "revised": revised_score}


DATASETS = {"GAIA": load_gaia, "Git-10M": load_git10m, "DDFAV": load_ddfav}


def bleu_cider_report(name: str) -> dict:
    """Compute the BLEU/CIDEr report for one dataset (``GAIA`` / ``Git-10M`` / ``DDFAV``)."""
    loader = DATASETS[name]
    samples = [s for s in loader() if s["refs"] and s["original"] and s["revised"]]
    return {
        "name": name,
        "num_samples": len(samples),
        "bleu": compute_bleu(samples),
        "cider": compute_cider(samples),
    }
