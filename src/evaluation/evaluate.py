from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from datasets import load_dataset
from jsonschema import ValidationError, validate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel

try:
    import mlflow
except ImportError:  # pragma: no cover - optional runtime dependency
    mlflow = None

from src.evaluation.metrics import EvalMetrics
from src.utils.config import load_yaml
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation harness for structured JSON generation")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--schema-config", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--eval-data-path", default=None)
    return parser.parse_args()


def build_model_kwargs(model_cfg: dict) -> dict:
    kwargs = {
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
    }
    if model_cfg.get("load_in_4bit", False):
        quant_cfg = model_cfg.get("quantization", {})
        kwargs["quantization_config"] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            "bnb_4bit_compute_dtype": getattr(
                torch,
                quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"),
            ),
            "bnb_4bit_use_double_quant": quant_cfg.get("bnb_4bit_use_double_quant", True),
        }
        kwargs["device_map"] = model_cfg.get("device_map", "auto")
    return kwargs


def build_prompt(instruction: str, user_input: str) -> str:
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Input:\n"
        f"{user_input}\n\n"
        "### Response:\n"
    )


def required_fields_complete(payload: dict, required_fields: list[str]) -> bool:
    return all(field in payload and payload[field] not in (None, "", []) for field in required_fields)


def main() -> None:
    args = parse_args()
    setup_logging()

    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.training_config)
    schema_cfg = load_yaml(args.schema_config)

    if mlflow:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(train_cfg["mlflow"]["experiment_name"])

    adapter_path = args.adapter_path or (train_cfg["output_dir"] + "/adapter")

    tokenizer_path = adapter_path if Path(adapter_path).exists() else model_cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model_kwargs = build_model_kwargs(model_cfg)
    if "quantization_config" in model_kwargs:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(**model_kwargs["quantization_config"])
    model = AutoModelForCausalLM.from_pretrained(model_cfg["model_name"], **model_kwargs)

    if Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
        LOGGER.info("Loaded adapter from %s", adapter_path)
    else:
        LOGGER.info("Adapter not found at %s; evaluating base model only.", adapter_path)

    eval_data_path = args.eval_data_path or train_cfg["dataset"]["eval_path"]
    eval_data = load_dataset("json", data_files=eval_data_path, split="train")
    schema = schema_cfg["schema"]
    required_fields = schema.get("required", [])

    metrics = EvalMetrics(total=len(eval_data))
    detailed = []

    for row in eval_data:
            prompt = build_prompt(row["instruction"], row.get("input", ""))
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generated = model.generate(**inputs, max_new_tokens=train_cfg.get("eval_max_new_tokens", 256))
            decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
            candidate = decoded.split("### Response:\n")[-1].strip()

            result = {
                "instruction": row["instruction"],
                "expected": row["output"],
                "prediction": candidate,
                "json_parse": False,
                "schema_valid": False,
                "required_complete": False,
                "error": None,
            }

            try:
                parsed = json.loads(candidate)
                result["json_parse"] = True
                metrics.json_parse_success += 1
                try:
                    validate(instance=parsed, schema=schema)
                    result["schema_valid"] = True
                    metrics.schema_valid += 1
                except ValidationError as schema_err:
                    result["error"] = f"schema_error: {schema_err.message}"

                if required_fields_complete(parsed, required_fields):
                    result["required_complete"] = True
                    metrics.required_complete += 1
            except json.JSONDecodeError as json_err:
                result["error"] = f"json_error: {str(json_err)}"

            detailed.append(result)

    report = {
        "metrics": metrics.to_dict(),
        "samples": detailed,
    }

    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if mlflow:
        with mlflow.start_run(run_name=train_cfg["mlflow"].get("eval_run_name", "qwen-qlora-eval")):
            for key, value in report["metrics"].items():
                if isinstance(value, (float, int)):
                    mlflow.log_metric(key, value)

            mlflow.log_artifact(str(output_report), artifact_path="evaluation")
    else:
        LOGGER.warning("mlflow is not installed; skipping experiment tracking.")

    LOGGER.info("Evaluation report written to %s", output_report)


if __name__ == "__main__":
    main()
