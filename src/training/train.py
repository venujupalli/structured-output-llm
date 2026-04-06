from __future__ import annotations

import argparse
import inspect
import logging
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from src.data.dataset import format_example, load_alpaca_dataset
from src.utils.config import deep_merge, load_yaml
from src.utils.logging_utils import setup_logging
from src.utils.mlflow_utils import configure_mlflow_paths, safe_log_artifacts
from src.utils.runtime import log_accelerator_report, recommend_model_name, resolve_adapter_mode, resolve_device

LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[2]


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


def build_sft_trainer(
    *,
    model,
    train_dataset,
    peft_config: LoraConfig,
    tokenizer,
    training_args: TrainingArguments,
    training_cfg: dict,
) -> SFTTrainer:
    signature = inspect.signature(SFTTrainer.__init__)
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
        "args": training_args,
    }

    if "tokenizer" in signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    if "dataset_text_field" in signature.parameters:
        trainer_kwargs["dataset_text_field"] = "text"

    if "max_seq_length" in signature.parameters:
        trainer_kwargs["max_seq_length"] = training_cfg["max_seq_length"]
    elif "max_length" in signature.parameters:
        trainer_kwargs["max_length"] = training_cfg["max_seq_length"]

    if "packing" in signature.parameters:
        trainer_kwargs["packing"] = training_cfg.get("packing", False)

    if "dataset_kwargs" in signature.parameters:
        trainer_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": False}

    return SFTTrainer(**trainer_kwargs)


def normalize_training_config(training_cfg: dict) -> dict:
    normalized = dict(training_cfg)
    int_fields = [
        "seed",
        "max_seq_length",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "save_steps",
        "save_total_limit",
        "eval_max_new_tokens",
    ]
    float_fields = [
        "learning_rate",
        "warmup_ratio",
    ]
    bool_fields = [
        "bf16",
        "fp16",
        "packing",
    ]

    for field in int_fields:
        if field in normalized:
            normalized[field] = int(normalized[field])

    for field in float_fields:
        if field in normalized:
            normalized[field] = float(normalized[field])

    for field in bool_fields:
        if field in normalized:
            value = normalized[field]
            if isinstance(value, str):
                normalized[field] = value.strip().lower() in {"1", "true", "yes", "on"}
            else:
                normalized[field] = bool(value)

    dataset_cfg = dict(normalized.get("dataset", {}))
    if "train_max_records" in dataset_cfg:
        dataset_cfg["train_max_records"] = int(dataset_cfg["train_max_records"])
    normalized["dataset"] = dataset_cfg

    return normalized


def main() -> None:
    args = parse_args()
    setup_logging()

    model_cfg = load_yaml(args.model_config)
    training_cfg = load_yaml(args.training_config)
    if args.override_config:
        training_cfg = deep_merge(training_cfg, load_yaml(args.override_config))
    training_cfg = normalize_training_config(training_cfg)
    model_cfg = dict(model_cfg)
    model_cfg["model_name"] = recommend_model_name(model_cfg)
    model_cfg["adapter_mode"] = resolve_adapter_mode(model_cfg)
    model_cfg["device_map"] = resolve_device(model_cfg)
    log_accelerator_report(LOGGER, model_cfg, context="Training")

    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)

    configure_mlflow_paths(
        mlflow,
        training_cfg["mlflow"]["experiment_name"],
        root_dir=ROOT_DIR,
        logger=LOGGER,
        tracking_uri=training_cfg["mlflow"].get("tracking_uri"),
        artifact_root=training_cfg["mlflow"].get("artifact_root"),
    )

    with mlflow.start_run(run_name=training_cfg["mlflow"].get("run_name", "qwen-qlora-train")):
        mlflow.log_params(
            {
                "base_model": model_cfg["model_name"],
                "adapter_mode": model_cfg["adapter_mode"],
                "training_preset": training_cfg.get("preset_name", "custom"),
                "learning_rate": training_cfg["learning_rate"],
                "batch_size": training_cfg["per_device_train_batch_size"],
                "gradient_accumulation_steps": training_cfg["gradient_accumulation_steps"],
                "lora_r": model_cfg["lora"]["r"],
                "lora_alpha": model_cfg["lora"]["lora_alpha"],
                "lora_dropout": model_cfg["lora"]["lora_dropout"],
                "seed": seed,
            }
        )

        bnb_config = build_quantization_config(model_cfg) if model_cfg["adapter_mode"] == "qlora" else None

        tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"], use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": model_cfg.get("trust_remote_code", False),
        }
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = model_cfg.get("device_map", "auto")
        elif model_cfg.get("device_map") == "mps":
            model_kwargs["device_map"] = {"": "mps"}
        elif torch.cuda.is_available():
            model_kwargs["device_map"] = model_cfg.get("device_map", "auto")

        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_name"],
            **model_kwargs,
        )
        if model_cfg["adapter_mode"] == "qlora":
            model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

        peft_config = LoraConfig(
            r=model_cfg["lora"]["r"],
            lora_alpha=model_cfg["lora"]["lora_alpha"],
            lora_dropout=model_cfg["lora"]["lora_dropout"],
            target_modules=model_cfg["lora"]["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        train_dataset = load_alpaca_dataset(training_cfg["dataset"]["train_path"], split="train")
        train_max_records = training_cfg["dataset"].get("train_max_records")
        if train_max_records:
            train_dataset = train_dataset.select(range(min(train_max_records, len(train_dataset))))
            LOGGER.info("Training dataset limited to %d records", len(train_dataset))
        train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
        LOGGER.info("Training preset: %s", training_cfg.get("preset_name", "custom"))

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
            dataloader_pin_memory=torch.cuda.is_available(),
            report_to=[],
            seed=seed,
        )

        trainer = build_sft_trainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            tokenizer=tokenizer,
            training_args=training_args,
            training_cfg=training_cfg,
        )

        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.model.save_pretrained(str(output_dir / "adapter"))
        tokenizer.save_pretrained(str(output_dir / "adapter"))

        safe_log_artifacts(mlflow, str(output_dir / "adapter"), artifact_path="adapter", logger=LOGGER)
        mlflow.log_metric("train_samples", len(train_dataset))
        LOGGER.info("Training completed and adapter saved.")


if __name__ == "__main__":
    main()
