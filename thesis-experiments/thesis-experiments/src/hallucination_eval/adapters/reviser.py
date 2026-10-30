from __future__ import annotations

from typing import Any

from hallucination_eval.adapters.base import Processor, revise_prompt_text

CONV_MODE = "llava_v1"


class ReviserAdapter:
    def __init__(
        self,
        base_path: str = "/mnt/storagecube/filipe/models/rs-llava-merged",
        lora_path: str = "/mnt/storagecube/filipe/out/results/rs-llava-lure-lora",
        conv_mode: str = CONV_MODE,
    ) -> None:
        self.base_path = base_path
        self.lora_path = lora_path
        self.conv_mode = conv_mode

    def load(self) -> tuple[Any, Processor]:
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from peft import PeftModel

        disable_torch_init()
        model_name = get_model_name_from_path(self.base_path)
        print(f"Loading base model  : {self.base_path}")
        tokenizer, model, image_processor, _context_len = load_pretrained_model(
            self.base_path, None, model_name
        )

        print(f"Applying LoRA adapter: {self.lora_path}")
        model = PeftModel.from_pretrained(model, self.lora_path)

        # Left padding is required for batched generation in causal LLMs.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token

        return model, Processor(tokenizer, image_processor)

    def build_prompt(self, caption: str, model: Any) -> str:
        from llava.constants import (
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
        )

        text = revise_prompt_text(caption)
        if model.config.mm_use_im_start_end:
            return (
                f"{DEFAULT_IM_START_TOKEN} {DEFAULT_IMAGE_TOKEN} {DEFAULT_IM_END_TOKEN}\n"
                f"{text}"
            )
        return f"{DEFAULT_IMAGE_TOKEN}\n{text}"

    def generate_batch(
        self,
        items: list[tuple[Any, str]],
        model: Any,
        processor: Processor,
        max_new_tokens: int = 256,
    ) -> list[str]:
        """Revise a batch of (image, caption) pairs with greedy decoding"""
        import torch

        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        batch_ids: list[Any] = []
        batch_images: list[Any] = []

        for image, caption in items:
            img_tensor = processor.image_processor.preprocess(
                image.convert("RGB"), return_tensors="pt"
            )["pixel_values"][0]

            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], self.build_prompt(caption, model))
            conv.append_message(conv.roles[1], None)
            input_ids = tokenizer_image_token(
                conv.get_prompt(), processor.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            batch_ids.append(input_ids)
            batch_images.append(img_tensor)

        if not batch_ids:
            return []

        squeezed = [ids.squeeze(0) for ids in batch_ids]
        padded = processor.tokenizer.pad(
            {"input_ids": squeezed}, padding=True, return_tensors="pt"
        )
        batched_ids = padded["input_ids"].cuda()
        attention_mask = padded["attention_mask"].cuda()
        batched_images = torch.stack(batch_images).half().cuda()

        input_len = batched_ids.shape[1]
        with torch.inference_mode():
            output_ids = model.generate(
                inputs=batched_ids,
                images=batched_images,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy decoding for determinism in revision
                use_cache=True,
            )

        return [
            text.strip()
            for text in processor.tokenizer.batch_decode(
                output_ids[:, input_len:], skip_special_tokens=True
            )
        ]
