from __future__ import annotations

from typing import Any

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from hallucination_eval.adapters.base import Processor

CONV_MODE = "llava_v1"


class LlavaFamilyAdapter:
    """Shared LLaVA-1.5 / RS-LLaVA adapter"""

    answer_token_forms = {"A": ["A", "▁A"], "B": ["B", "▁B"]}

    def __init__(self, model_id: str, model_base: str | None = None) -> None:
        self.model_id = model_id
        self.model_base = model_base

    def load(self) -> tuple[Any, Processor]:
        disable_torch_init()
        model_name = get_model_name_from_path(self.model_id)
        print(f"Loading model  : {model_name}")
        print(f"Loading base   : {self.model_base}")
        tokenizer, model, image_processor, _context_len = load_pretrained_model(
            self.model_id, self.model_base, model_name
        )
        return model, Processor(tokenizer, image_processor)

    def tokenizer(self, processor: Processor) -> Any:
        return processor.tokenizer

    def _preprocess(self, image: Any, processor: Processor) -> Any:
        return processor.image_processor.preprocess(image.convert("RGB"), return_tensors="pt")[
            "pixel_values"
        ][0]

    def _build_prompt(self, question: str, model: Any) -> str:
        if model.config.mm_use_im_start_end:
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
        """Generate ``num_return_sequences`` free-form answers (§5.3 sampling)."""
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

        if model.config.mm_use_im_start_end:
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


class Llava15Adapter(LlavaFamilyAdapter):
    mcq_token_prefixes = ["", "▁", " "]

    def __init__(self) -> None:
        super().__init__("liuhaotian/llava-v1.5-7b", None)


class RsLlavaAdapter(LlavaFamilyAdapter):
    mcq_token_prefixes = ["", "▁"]

    def __init__(self) -> None:
        super().__init__("BigData-KSU/RS-llava-v1.5-7b-LoRA", "Intel/neural-chat-7b-v3-3")
