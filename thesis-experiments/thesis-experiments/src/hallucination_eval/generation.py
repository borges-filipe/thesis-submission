from __future__ import annotations

import torch
from transformers import LogitsProcessor


class APCLogitsProcessor(LogitsProcessor):
    """Adaptive Plausibility Constraint"""

    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.beta > 0.0:
            probs = torch.softmax(scores, dim=-1)
            max_probs = probs.max(dim=-1, keepdim=True).values
            apc_mask = probs < (self.beta * max_probs)
            scores[apc_mask] = -float("inf")
        return scores


class VCDLogitsProcessor(LogitsProcessor):
    """Visual contrastive decoding"""

    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[0] != 2:
            return scores

        logits_orig = scores[0:1]
        logits_dist = scores[1:2]

        probs_orig = torch.softmax(logits_orig, dim=-1)
        log_p_orig = torch.log_softmax(logits_orig, dim=-1)
        log_p_dist = torch.log_softmax(logits_dist, dim=-1)

        vcd_logits = (1.0 + self.alpha) * log_p_orig - self.alpha * log_p_dist

        if self.beta > 0.0:
            max_probs = probs_orig.max(dim=-1, keepdim=True).values
            apc_mask = probs_orig < (self.beta * max_probs)
            vcd_logits[apc_mask] = -float("inf")

        scores[0] = vcd_logits.squeeze(0)
        scores[1] = vcd_logits.squeeze(0)
        return scores
