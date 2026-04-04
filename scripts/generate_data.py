#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generator import OrderDataGenerator, save_json, save_jsonl
from src.utils.config import load_yaml
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic and golden datasets for order extraction")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    return parser.parse_args()


def split_train_val(rows: list[dict[str, str]], val_ratio: float) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    split_idx = int(len(rows) * (1 - val_ratio))
    return rows[:split_idx], rows[split_idx:]


def main() -> None:
    args = parse_args()
    setup_logging()

    data_cfg = load_yaml(args.data_config)
    generation_cfg = data_cfg["generation"]
    paths_cfg = data_cfg["paths"]

    schema_cfg = load_yaml(paths_cfg["schema_config"])
    schema = schema_cfg["schema"]

    generator = OrderDataGenerator(schema=schema, config=data_cfg)

    synthetic = generator.generate_synthetic_dataset(
        n_samples=generation_cfg["synthetic_samples"],
        batch_size=generation_cfg["batch_size"],
        max_retries=generation_cfg["max_retries"],
    )
    train_rows, val_rows = split_train_val(synthetic, val_ratio=generation_cfg["val_ratio"])

    golden_rows = generator.generate_golden_dataset(generation_cfg["golden_samples"])

    save_jsonl(paths_cfg["synthetic_train_jsonl"], train_rows)
    save_jsonl(paths_cfg["synthetic_val_jsonl"], val_rows)
    save_jsonl(paths_cfg["golden_jsonl"], golden_rows)

    if generation_cfg.get("save_json_copy", False):
        save_json(paths_cfg["synthetic_train_json"], train_rows)
        save_json(paths_cfg["synthetic_val_json"], val_rows)
        save_json(paths_cfg["golden_json"], golden_rows)

    LOGGER.info("Synthetic train samples: %s", len(train_rows))
    LOGGER.info("Synthetic val samples: %s", len(val_rows))
    LOGGER.info("Golden samples: %s", len(golden_rows))
    LOGGER.info("Data generation completed successfully.")


if __name__ == "__main__":
    main()
