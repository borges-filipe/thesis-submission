"""Model adapters"""

from hallucination_eval.adapters.base import (
    BinaryLogits,
    Processor,
    VLMAdapter,
    adapter_binary_logits,
    binary_question,
    confidence_visible,
    extract_binary_logits,
)

__all__ = [
    "BinaryLogits",
    "Processor",
    "VLMAdapter",
    "adapter_binary_logits",
    "binary_question",
    "confidence_visible",
    "extract_binary_logits",
]
