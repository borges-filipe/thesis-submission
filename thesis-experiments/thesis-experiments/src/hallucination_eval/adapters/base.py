"""Adapter helpers"""

from __future__ import annotations

from dataclasses import dataclass
from string import ascii_uppercase
from typing import Any, Protocol, runtime_checkable

from hallucination_eval.decoding import sigmoid

MCQ_STEM = "What object is visible in the provided image?"


@dataclass
class BinaryLogits:
    """Result of a binary (A/B) next-token logit probe."""

    logits: dict[str, float]  # {"A": ..., "B": ...}
    tokens: dict[str, str]  # {"A": "▁A", "B": "▁B"} (the chosen token form)


@dataclass
class Processor:
    """Minimal namespace wrapping a tokenizer + image processor.

    Used by the LLaVA-family and GeoChat adapters (whose upstream loaders
    return these two objects together).
    """

    tokenizer: Any
    image_processor: Any


@runtime_checkable
class VLMAdapter(Protocol):
    """1 instance per model"""

    model_id: str
    answer_token_forms: dict[str, list[str]]  # {"A": [...], "B": [...]}
    mcq_token_prefixes: list[str]  # e.g. ["", "▁", " "] for the MCQ letters

    def load(self) -> tuple[Any, Any]:
        """Return (model, processor)"""
        ...

    def tokenizer(self, processor: Any) -> Any:
        """Return tokenizer """
        ...

    def forward_logits(self, question: str, image: Any, model: Any, processor: Any) -> Any:
        """Run one forward pass and return the logit vector"""
        ...


def binary_question(text: str) -> str:
    """Format the binary visibility question"""
    return f"{text}\nA) Yes\nB) No\nLetter:"


def single_word_question(text: str) -> str:
    """Format the free-form (single-word) generation question"""
    return f"{text} Please answer this question with one word."


def parse_yes_no(user_input: str) -> bool:
    """Parse a generated yes/no answer into a boolean"""
    normalized = user_input.strip().lower()
    if normalized in ("yes", "y"):
        return True
    if normalized in ("no", "n"):
        return False
    raise ValueError(f"Input must be 'yes' or 'no'. Received: {user_input!r}")


# The exact instruction used for the §5.4 LURE caption reviser (training + inference).
REVISER_INSTRUCTION = (
    "You are a remote sensing image caption reviser. Revise the following image description to ensure "
    "it strictly matches the visual content. First, resolve any \"[IDK]\" placeholders by replacing them "
    "with the correct objects from the image or deleting them entirely. Second, verify all other explicitly "
    "named objects in the text and remove any items that are not physically present in the visual scene."
)


def revise_prompt_text(caption: str) -> str:
    """The textual part of the reviser prompt"""
    return f"{REVISER_INSTRUCTION}\n\nDescription to revise: {caption}"


def mcq_question(options: list[str], stem: str = MCQ_STEM) -> str:
    """Format the multiple-choice question"""
    if len(options) > len(ascii_uppercase):
        raise ValueError(
            f"Too many options ({len(options)}). Maximum supported is {len(ascii_uppercase)}."
        )
    lines = [stem, ""]
    for i, option in enumerate(options):
        lines.append(f"{ascii_uppercase[i]}) {option}")
    lines.extend(["", "Letter:"])
    return "\n".join(lines)


def confidence_visible(logits: dict[str, float]) -> float:
    """P(object visible) = sigmoid(logit_A - logit_B)."""
    return sigmoid(logits.get("A", 0.0) - logits.get("B", 0.0))


def extract_binary_logits(
    next_token_logits: Any,
    tokenizer: Any,
    forms: dict[str, list[str]],
) -> BinaryLogits:
    """Pull the A/B logits out of a next-token logit vector"""
    logits: dict[str, float] = {}
    tokens: dict[str, str] = {}

    unk = getattr(tokenizer, "unk_token_id", None)
    for letter, fmts in forms.items():
        token_ids = [
            tid
            for tid in tokenizer.convert_tokens_to_ids(fmts)
            if tid is not None and tid != unk
        ]
        if not token_ids:
            token_ids = [tokenizer.convert_tokens_to_ids(letter)]
        best_tid = max(token_ids, key=lambda tid: next_token_logits[tid].item())
        logits[letter] = next_token_logits[best_tid].item()
        tokens[letter] = tokenizer.convert_ids_to_tokens(best_tid)

    return BinaryLogits(logits=logits, tokens=tokens)


def adapter_binary_logits(
    adapter: VLMAdapter,
    question: str,
    image: Any,
    model: Any,
    processor: Any,
) -> BinaryLogits:
    """extract the A/B logits."""
    return extract_binary_logits(
        adapter.forward_logits(question, image, model, processor),
        adapter.tokenizer(processor),
        adapter.answer_token_forms,
    )


def extract_mcq_logits(
    next_token_logits: Any,
    tokenizer: Any,
    letters: list[str],
    prefixes: list[str],
) -> tuple[dict[str, float], dict[str, str]]:
    """Pull the per-letter logits out of a next-token logit vector."""
    logits: dict[str, float] = {}
    tokens: dict[str, str] = {}

    unk = getattr(tokenizer, "unk_token_id", None)
    for letter in letters:
        forms = [f"{prefix}{letter}" for prefix in prefixes]
        token_ids = [
            tid
            for tid in tokenizer.convert_tokens_to_ids(forms)
            if tid is not None and tid != unk
        ]
        if not token_ids:
            fallback = tokenizer.convert_tokens_to_ids(letter)
            if fallback is not None and fallback != unk:
                token_ids = [fallback]
            else:
                continue

        best_tid = max(token_ids, key=lambda tid: next_token_logits[tid].item())
        logits[letter] = next_token_logits[best_tid].item()
        token_str = tokenizer.convert_ids_to_tokens(best_tid)
        tokens[letter] = token_str if isinstance(token_str, str) else str(token_str)

    return logits, tokens


def adapter_mcq_logits(
    adapter: VLMAdapter,
    options: list[str],
    image: Any,
    model: Any,
    processor: Any,
) -> tuple[dict[str, float], dict[str, str]]:
    """Run one MCQ forward, give options and extract the letter logits"""
    question = mcq_question(options)
    forward = getattr(adapter, "mcq_forward_logits", None)
    if forward is None:
        forward = adapter.forward_logits
    next_token_logits = forward(question, image, model, processor)
    letters = [ascii_uppercase[i] for i in range(len(options))]
    return extract_mcq_logits(
        next_token_logits,
        adapter.tokenizer(processor),
        letters,
        adapter.mcq_token_prefixes,
    )
