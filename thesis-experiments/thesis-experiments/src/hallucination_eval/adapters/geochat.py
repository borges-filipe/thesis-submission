from __future__ import annotations

from typing import Any

from geochat.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from geochat.conversation import conv_templates
from geochat.mm_utils import tokenizer_image_token
from geochat.utils import disable_torch_init

from hallucination_eval.adapters.base import Processor

MODEL_ID = "MBZUAI/geochat-7B"
CONV_MODE = "vicuna_v1"


class GeoChatAdapter:
    answer_token_forms = {"A": ["A", "▁A"], "B": ["B", "▁B"]}
    mcq_token_prefixes = ["", "▁"]

    def __init__(self) -> None:
        self.model_id = MODEL_ID

    def load(self) -> tuple[Any, Processor]:
        from geochat.model import GeoChatLlamaForCausalLM
        from transformers import AutoTokenizer

        disable_torch_init()
        print(f"Loading model  : {self.model_id}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)

        # Load in full precision first — the CLIP vision tower's bicubic
        # interpolate doesn't support fp16 in PyTorch 2.0.1.
        model = GeoChatLlamaForCausalLM.from_pretrained(
            self.model_id,
            ignore_mismatched_sizes=True,  # Prevents vision tower shape mismatch errors
        )

        # The vision tower initialises with 504px position embeddings but the
        # image processor defaults to 336px; resize back to 336px in fp32.
        vision_tower = model.get_model().vision_tower
        vision_tower.clip_interpolate_embeddings(image_size=336, patch_size=14)

        model = model.half().cuda()
        image_processor = vision_tower.image_processor

        return model, Processor(tokenizer, image_processor)

    def tokenizer(self, processor: Processor) -> Any:
        return processor.tokenizer

    def _preprocess(self, image: Any, processor: Processor) -> Any:
        return processor.image_processor.preprocess(image.convert("RGB"), return_tensors="pt")[
            "pixel_values"
        ][0]

    def _build_prompt(self, question: str, model: Any) -> str:
        if getattr(model.config, "mm_use_im_start_end", False):
            return (
                f"{DEFAULT_IM_START_TOKEN} {DEFAULT_IMAGE_TOKEN} {DEFAULT_IM_END_TOKEN}\n"
                f"{question}"
            )
        return f"{DEFAULT_IMAGE_TOKEN}\n{question}"

    def _tokenize(self, prompt: str, processor: Processor) -> Any:
        return (
            tokenizer_image_token(
                prompt, processor.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

    def generate(
        self,
        question: str,
        images: Any,
        model: Any,
        processor: Processor,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float | None = None,
        max_new_tokens: int = 16,
        logits_processor: Any = None,
    ) -> str:
        import torch

        if isinstance(images, (list, tuple)):
            batch = torch.stack([self._preprocess(im, processor) for im in images], dim=0).half().cuda()
        else:
            batch = self._preprocess(images, processor).unsqueeze(0).half().cuda()

        prompt = self._build_prompt(question, model)
        conv = conv_templates[CONV_MODE].copy()
        conv.append_message(conv.roles[0], prompt)  # USER turn
        conv.append_message(conv.roles[1], None)  # ASSISTANT: (open)
        input_ids = self._tokenize(conv.get_prompt(), processor)
        if batch.shape[0] > 1:
            input_ids = input_ids.expand(batch.shape[0], -1)

        gen_kwargs = dict(
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if logits_processor is not None:
            gen_kwargs["logits_processor"] = logits_processor

        with torch.inference_mode():
            output_ids = model.generate(input_ids, images=batch, **gen_kwargs)

        input_len = input_ids.shape[1]
        return processor.tokenizer.batch_decode(
            output_ids[0:1, input_len:], skip_special_tokens=True
        )[0].strip()

    def sample(
        self,
        question: str,
        image: Any,
        model: Any,
        processor: Processor,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float | None = None,
        top_k: int | None = None,
        max_new_tokens: int = 50,
    ) -> list[str]:
        """Generate num_return_sequences free-form answers """
        import torch

        batch = self._preprocess(image, processor).unsqueeze(0).half().cuda()
        prompt = self._build_prompt(question, model)
        conv = conv_templates[CONV_MODE].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        input_ids = self._tokenize(conv.get_prompt(), processor)

        gen_kwargs = dict(
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            use_cache=True,
        )
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

        with torch.inference_mode():
            output_ids = model.generate(input_ids, images=batch, **gen_kwargs)

        input_len = input_ids.shape[1]
        return [
            processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in output_ids[:, input_len:]
        ]

    def forward_logits(self, question: str, image: Any, model: Any, processor: Processor) -> Any:
        import torch

        image_rgb = image.convert("RGB")
        image_tensor = processor.image_processor.preprocess(image_rgb, return_tensors="pt")[
            "pixel_values"
        ][0]

        # Safely check for mm_use_im_start_end in the custom config.
        if getattr(model.config, "mm_use_im_start_end", False):
            prompt_text = (
                f"{DEFAULT_IM_START_TOKEN} {DEFAULT_IMAGE_TOKEN} {DEFAULT_IM_END_TOKEN}\n"
                f"{question}"
            )
        else:
            prompt_text = f"{DEFAULT_IMAGE_TOKEN}\n{question}"

        conv = conv_templates[CONV_MODE].copy()
        conv.append_message(conv.roles[0], prompt_text)  # USER turn
        conv.append_message(conv.roles[1], None)  # ASSISTANT: (open)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, processor.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            outputs = model(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                return_dict=True,
            )

        return outputs.logits[0, -1, :]
