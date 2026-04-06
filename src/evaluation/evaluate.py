from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

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
from src.utils.mlflow_utils import configure_mlflow_paths, safe_log_artifact
from src.utils.runtime import (
    log_accelerator_report,
    recommend_model_name,
    resolve_adapter_mode,
    resolve_device,
)

LOGGER = logging.getLogger(__name__)
PROGRESS_LOG_INTERVAL = 25
DEFAULT_EVAL_BATCH_SIZE = 4


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
    if model_cfg.get("adapter_mode") == "qlora" and model_cfg.get("load_in_4bit", False):
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


def log_eval_progress(metrics: EvalMetrics, *, completed: int, total: int, started_at: float) -> None:
    elapsed = max(time.time() - started_at, 0.0)
    rate = completed / elapsed if elapsed else 0.0
    LOGGER.info(
        "Evaluation progress: %d/%d (%.1f%%) | elapsed=%.1fs | samples_per_second=%.2f | json_parse=%d | schema_valid=%d | required_complete=%d",
        completed,
        total,
        (completed / total * 100) if total else 100.0,
        elapsed,
        rate,
        metrics.json_parse_success,
        metrics.schema_valid,
        metrics.required_complete,
    )


def evaluate_candidate(candidate: str, row: dict, schema: dict, required_fields: list[str], metrics: EvalMetrics) -> dict:
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

    return result


def main() -> None:
    args = parse_args()
    setup_logging()

    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.training_config)
    schema_cfg = load_yaml(args.schema_config)
    model_cfg = dict(model_cfg)
    model_cfg["model_name"] = recommend_model_name(model_cfg)
    model_cfg["adapter_mode"] = resolve_adapter_mode(model_cfg)
    model_cfg["device_map"] = resolve_device(model_cfg)
    log_accelerator_report(LOGGER, model_cfg, context="Evaluation")

    if mlflow:
        configure_mlflow_paths(
            mlflow,
            train_cfg["mlflow"]["experiment_name"],
            root_dir=Path(__file__).resolve().parents[2],
            logger=LOGGER,
            tracking_uri=train_cfg["mlflow"].get("tracking_uri"),
            artifact_root=train_cfg["mlflow"].get("artifact_root"),
        )

    adapter_path = args.adapter_path or (train_cfg["output_dir"] + "/adapter")

    tokenizer_path = adapter_path if Path(adapter_path).exists() else model_cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_kwargs = build_model_kwargs(model_cfg)
    if "quantization_config" in model_kwargs:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(**model_kwargs["quantization_config"])
    elif model_cfg.get("device_map") == "mps":
        model_kwargs["device_map"] = {"": "mps"}
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
    started_at = time.time()
    eval_batch_size = int(train_cfg.get("eval_batch_size", DEFAULT_EVAL_BATCH_SIZE))
    LOGGER.info(
        "Starting evaluation over %d samples from %s with adapter=%s batch_size=%d",
        metrics.total,
        eval_data_path,
        adapter_path if Path(adapter_path).exists() else "base-model-only",
        eval_batch_size,
    )
    next_progress_log = PROGRESS_LOG_INTERVAL

    for batch_start in range(0, metrics.total, eval_batch_size):
        batch_end = min(batch_start + eval_batch_size, metrics.total)
        batch_rows = [eval_data[idx] for idx in range(batch_start, batch_end)]
        prompts = [build_prompt(row["instruction"], row.get("input", "")) for row in batch_rows]
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=train_cfg.get("max_seq_length"),
        )
        prompt_lengths = tokenized["attention_mask"].sum(dim=1).tolist()
        inputs = {name: value.to(model.device) for name, value in tokenized.items()}
        generated = model.generate(**inputs, max_new_tokens=train_cfg.get("eval_max_new_tokens", 256))

        for row, sequence, prompt_length in zip(batch_rows, generated, prompt_lengths):
            completion_tokens = sequence[int(prompt_length):]
            candidate = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
            detailed.append(evaluate_candidate(candidate, row, schema, required_fields, metrics))

        completed = batch_end
        if completed >= next_progress_log or completed == metrics.total:
            log_eval_progress(metrics, completed=completed, total=metrics.total, started_at=started_at)
            while next_progress_log <= completed:
                next_progress_log += PROGRESS_LOG_INTERVAL

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

            safe_log_artifact(mlflow, str(output_report), artifact_path="evaluation", logger=LOGGER)
    else:
        LOGGER.warning("mlflow is not installed; skipping experiment tracking.")

    LOGGER.info("Evaluation report written to %s", output_report)


if __name__ == "__main__":
    main()
