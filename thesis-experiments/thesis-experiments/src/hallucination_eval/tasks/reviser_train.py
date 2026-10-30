from __future__ import annotations

import argparse
import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import PIL.Image
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

CONV_MODE = "llava_v1"


class LossPlottingCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def on_train_end(self, args, state, control, **kwargs):
        steps = []
        losses = []
        for log in state.log_history:
            if "loss" in log and "step" in log:
                steps.append(log["step"])
                losses.append(log["loss"])

        if steps and losses:
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses, label="Training Loss", color="#1f77b4", linewidth=2)
            plt.xlabel("Training Steps", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.title("Training Error over Time", fontsize=14)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, "training_loss_curve.png"), bbox_inches="tight")
            plt.close()
            print(f"Training loss curve saved to: {self.output_dir}")


class LUREDataset(Dataset):
    def __init__(self, data_path, image_folder, tokenizer, image_processor, conv_mode=CONV_MODE):
        super().__init__()
        with open(data_path, encoding="utf-8") as f:
            self.list_data_dict = json.load(f)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_mode = conv_mode

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        sources = self.list_data_dict[i]
        image = PIL.Image.open(os.path.join(self.image_folder, sources["image"])).convert("RGB")
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        human_msg = sources["conversations"][0]["value"]
        gpt_msg = sources["conversations"][1]["value"]

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], human_msg)
        conv.append_message(conv.roles[1], None)
        prompt_str = conv.get_prompt()

        conv.messages[-1][1] = gpt_msg
        full_str = conv.get_prompt()

        prompt_ids = tokenizer_image_token(
            prompt_str, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        full_ids = tokenizer_image_token(
            full_str, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        labels = full_ids.clone()
        labels[: len(prompt_ids)] = -100  # loss only on the revised caption

        return {"input_ids": full_ids, "labels": labels, "image": image_tensor}


class LUREDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, instances):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [i["input_ids"] for i in instances], batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [i["labels"] for i in instances], batch_first=True, padding_value=-100
        )
        images = torch.stack([i["image"] for i in instances])
        attention_mask = input_ids.ne(self.pad_token_id)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "images": images.half(),
        }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="reviser-train", description="Fine-tune the LURE reviser.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--merged", default="/mnt/storagecube/filipe/models/rs-llava-merged")
    parser.add_argument("--out", default="/mnt/storagecube/filipe/out/results/rs-llava-lure-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)

    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from peft import LoraConfig, get_peft_model

    disable_torch_init()
    print(f"Loading merged RS-LLaVA model from: {args.merged}")
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        args.merged, None, get_model_name_from_path(args.merged)
    )

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    train_dataset = LUREDataset(
        args.data, args.images, tokenizer, image_processor, conv_mode=CONV_MODE
    )
    data_collator = LUREDataCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[LossPlottingCallback(args.out)],
    )

    print("Starting LURE Fine-Tuning...")
    trainer.train()

    print(f"Saving fine-tuned LoRA weights to: {args.out}")
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print("Training complete!")


if __name__ == "__main__":
    main()
