from __future__ import annotations

from datasets import load_dataset


def load_alpaca_dataset(dataset_path: str, split: str = "train"):
    return load_dataset("json", data_files=dataset_path, split=split)


def format_example(example: dict) -> dict:
    prompt = (
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Input:\n"
        f"{example.get('input', '')}\n\n"
        "### Response:\n"
        f"{example['output']}"
    )
    return {"text": prompt}
