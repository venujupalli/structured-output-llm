from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from src.data.dataset import format_example, load_alpaca_dataset
from src.utils.config import deep_merge, load_yaml
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA training pipeline")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--override-config", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def build_quantization_config(model_cfg: dict) -> BitsAndBytesConfig | None:
    if not model_cfg.get("load_in_4bit", False):
        return None

    quant_cfg = model_cfg.get("quantization", {})
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(
            torch,
            quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"),
        ),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )


def main() -> None:
    args = parse_args()
    setup_logging()

    model_cfg = load_yaml(args.model_config)
    training_cfg = load_yaml(args.training_config)
    if args.override_config:
        training_cfg = deep_merge(training_cfg, load_yaml(args.override_config))

    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(training_cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=training_cfg["mlflow"].get("run_name", "qwen-qlora-train")):
        mlflow.log_params(
            {
                "base_model": model_cfg["model_name"],
                "learning_rate": training_cfg["learning_rate"],
                "batch_size": training_cfg["per_device_train_batch_size"],
                "gradient_accumulation_steps": training_cfg["gradient_accumulation_steps"],
                "lora_r": model_cfg["lora"]["r"],
                "lora_alpha": model_cfg["lora"]["lora_alpha"],
                "lora_dropout": model_cfg["lora"]["lora_dropout"],
                "seed": seed,
            }
        )

        bnb_config = build_quantization_config(model_cfg)

        tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"], use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": model_cfg.get("trust_remote_code", False),
        }
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = model_cfg.get("device_map", "auto")
        elif torch.cuda.is_available():
            model_kwargs["device_map"] = model_cfg.get("device_map", "auto")

        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_name"],
            **model_kwargs,
        )

        peft_config = LoraConfig(
            r=model_cfg["lora"]["r"],
            lora_alpha=model_cfg["lora"]["lora_alpha"],
            lora_dropout=model_cfg["lora"]["lora_dropout"],
            target_modules=model_cfg["lora"]["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        train_dataset = load_alpaca_dataset(training_cfg["dataset"]["train_path"], split="train")
        train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)

        output_dir = Path(training_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=training_cfg["num_train_epochs"],
            per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
            learning_rate=training_cfg["learning_rate"],
            logging_steps=training_cfg["logging_steps"],
            save_steps=training_cfg["save_steps"],
            save_total_limit=training_cfg["save_total_limit"],
            bf16=training_cfg.get("bf16", False),
            fp16=training_cfg.get("fp16", False),
            optim=training_cfg.get("optimizer", "paged_adamw_8bit"),
            lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
            warmup_ratio=training_cfg.get("warmup_ratio", 0.03),
            report_to=[],
            seed=seed,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=training_cfg["max_seq_length"],
            tokenizer=tokenizer,
            args=training_args,
            packing=training_cfg.get("packing", False),
        )

        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.model.save_pretrained(str(output_dir / "adapter"))
        tokenizer.save_pretrained(str(output_dir / "adapter"))

        mlflow.log_artifacts(str(output_dir / "adapter"), artifact_path="adapter")
        mlflow.log_metric("train_samples", len(train_dataset))
        LOGGER.info("Training completed and adapter saved.")


if __name__ == "__main__":
    main()
