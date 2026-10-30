from __future__ import annotations

from typing import Any

from hallucination_eval.config import HF_CACHE_DIR

QWEN_TOKEN_FORMS = {
    "A": ["A", "▁A", " A", "ĠA"],
    "B": ["B", "▁B", " B", "ĠB"],
}

_MISSING = object()


class _QwenAdapter:
    answer_token_forms = QWEN_TOKEN_FORMS
    mcq_token_prefixes = ["", "▁", " ", "Ġ"]

    def __init__(
        self,
        model_id: str,
        model_cls: Any,
        model_kwargs: dict[str, Any],
        processor_kwargs: dict[str, Any],
        enable_thinking: Any = _MISSING,
    ) -> None:
        self.model_id = model_id
        self._model_cls = model_cls
        self._model_kwargs = model_kwargs
        self._processor_kwargs = processor_kwargs
        self._enable_thinking = enable_thinking

    def load(self) -> tuple[Any, Any]:
        from transformers import AutoProcessor

        print(f"Loading model  : {self.model_id}")
        model = self._model_cls.from_pretrained(self.model_id, **self._model_kwargs)
        processor = AutoProcessor.from_pretrained(
            self.model_id, **self._processor_kwargs
        )
        return model, processor

    def tokenizer(self, processor: Any) -> Any:
        return processor.tokenizer

    def _apply_chat_template(self, question: str, image: Any, processor: Any) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._enable_thinking is not _MISSING:
            template_kwargs["enable_thinking"] = self._enable_thinking
        return processor.apply_chat_template(messages, **template_kwargs)

    def _build_inputs(
        self, question: str, image: Any, model: Any, processor: Any
    ) -> dict:
        text = self._apply_chat_template(question, image, processor)
        return processor(text=[text], images=[image], return_tensors="pt").to(
            model.device
        )

    def chat_generate(
        self,
        messages: list[dict],
        images: Any,
        model: Any,
        processor: Any,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate from a full (multi-turn) chat message list"""
        import torch

        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._enable_thinking is not _MISSING:
            template_kwargs["enable_thinking"] = self._enable_thinking
        text = processor.apply_chat_template(messages, **template_kwargs)

        if images:
            inputs = processor(text=[text], images=images, return_tensors="pt").to(
                model.device
            )
        else:
            inputs = processor(text=[text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        input_len = inputs["input_ids"].shape[1]
        return processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def generate(
        self,
        question: str,
        images: Any,
        model: Any,
        processor: Any,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float | None = None,
        max_new_tokens: int = 16,
        logits_processor: Any = None,
    ) -> str:
        import torch

        gen_kwargs: dict[str, Any] = dict(
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if logits_processor is not None:
            gen_kwargs["logits_processor"] = logits_processor

        if isinstance(images, (list, tuple)):
            # VCD: batch the original + distorted inputs with the same prompt.
            text = self._apply_chat_template(question, images[0], processor)
            inputs_orig = processor(
                text=[text], images=[images[0]], return_tensors="pt"
            ).to(model.device)
            inputs_dist = processor(
                text=[text], images=[images[1]], return_tensors="pt"
            ).to(model.device)
            gen_kwargs["input_ids"] = torch.cat(
                [inputs_orig["input_ids"], inputs_dist["input_ids"]], dim=0
            )
            gen_kwargs["attention_mask"] = torch.cat(
                [inputs_orig["attention_mask"], inputs_dist["attention_mask"]], dim=0
            )
            gen_kwargs["pixel_values"] = torch.cat(
                [inputs_orig["pixel_values"], inputs_dist["pixel_values"]], dim=0
            )
            if "image_grid_thw" in inputs_orig:
                gen_kwargs["image_grid_thw"] = torch.cat(
                    [inputs_orig["image_grid_thw"], inputs_dist["image_grid_thw"]],
                    dim=0,
                )
        else:
            gen_kwargs.update(self._build_inputs(question, images, model, processor))

        with torch.no_grad():
            output_ids = model.generate(**gen_kwargs)

        input_len = gen_kwargs["input_ids"].shape[1]
        return processor.tokenizer.batch_decode(
            output_ids[0:1, input_len:], skip_special_tokens=True
        )[0].strip()

    def sample(
        self,
        question: str,
        image: Any,
        model: Any,
        processor: Any,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float | None = None,
        top_k: int | None = None,
        max_new_tokens: int = 50,
    ) -> list[str]:
        """Generate num_return_sequences free-form answers"""
        import torch

        inputs = self._build_inputs(question, image, model, processor)
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

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        return [
            processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in output_ids[:, input_len:]
        ]

    def forward_logits(
        self, question: str, image: Any, model: Any, processor: Any
    ) -> Any:
        return self._forward(question, image, model, processor, self._enable_thinking)

    def mcq_forward_logits(
        self, question: str, image: Any, model: Any, processor: Any
    ) -> Any:
        return self._forward(question, image, model, processor, _MISSING)

    def _forward(
        self,
        question: str,
        image: Any,
        model: Any,
        processor: Any,
        enable_thinking: Any,
    ) -> Any:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not _MISSING:
            template_kwargs["enable_thinking"] = enable_thinking
        text = processor.apply_chat_template(messages, **template_kwargs)

        inputs = processor(text=[text], images=[image], return_tensors="pt").to(
            model.device
        )

        with torch.no_grad():
            outputs = model(**inputs)

        return outputs.logits[0, -1, :]


class Qwen3VLAdapter(_QwenAdapter):
    def __init__(self) -> None:
        import torch
        from transformers import Qwen3VLForConditionalGeneration

        super().__init__(
            "Qwen/Qwen3-VL-8B-Instruct",
            Qwen3VLForConditionalGeneration,
            model_kwargs=dict(
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            ),
            processor_kwargs=dict(cache_dir=str(HF_CACHE_DIR)),
        )


class Qwen35Adapter(_QwenAdapter):
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText

        super().__init__(
            "Qwen/Qwen3.5-9B",
            AutoModelForImageTextToText,
            model_kwargs=dict(
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                cache_dir=str(HF_CACHE_DIR),
                trust_remote_code=True,
            ),
            processor_kwargs=dict(
                cache_dir=str(HF_CACHE_DIR),
                trust_remote_code=True,
            ),
            enable_thinking=False,
        )
