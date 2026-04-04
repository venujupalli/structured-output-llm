from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from src.data.validator import validate_alpaca_sample

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptTemplates:
    instruction: str = (
        "Extract the e-commerce order information from the text and return a strict JSON object "
        "that matches the required schema."
    )


class OrderDataGenerator:
    def __init__(self, schema: dict[str, Any], config: dict[str, Any]) -> None:
        self.schema = schema
        self.config = config
        self.rng = random.Random(config["generation"]["seed"])
        self.prompt_templates = PromptTemplates()

        self.product_catalog = [
            "wireless mouse",
            "mechanical keyboard",
            "usb-c hub",
            "noise-cancelling headphones",
            "smart water bottle",
            "gaming monitor",
            "laptop stand",
            "desk lamp",
            "portable SSD",
            "webcam",
        ]
        self.customer_names = [
            "Avery Johnson",
            "Priya Shah",
            "Miguel Torres",
            "Harper Lee",
            "Jordan Kim",
            "Sofia Patel",
            "Noah Nguyen",
            "Emma Rodriguez",
        ]
        self.addresses = [
            "421 Pine Street, Seattle, WA 98101",
            "88 Lakeview Ave, Austin, TX 73301",
            "1524 Maple Drive, Denver, CO 80203",
            "740 Mission Blvd, San Diego, CA 92109",
            "19 Elm Court, Boston, MA 02108",
        ]
        self.styles = ["formal", "casual", "shorthand"]
        self.scenarios = [
            "simple",
            "multi_item",
            "ambiguous",
            "messy",
            "missing_partial",
            "incorrect_total",
            "repeated_items",
            "noisy",
        ]

    def _random_order_date(self) -> str:
        start = date(2024, 1, 1)
        offset = self.rng.randint(0, 730)
        return (start + timedelta(days=offset)).isoformat()

    def _build_items(self, scenario: str) -> list[dict[str, Any]]:
        if scenario == "simple":
            count = 1
        elif scenario in {"multi_item", "incorrect_total"}:
            count = self.rng.randint(2, 4)
        else:
            count = self.rng.randint(1, 3)

        items: list[dict[str, Any]] = []
        for _ in range(count):
            product = self.rng.choice(self.product_catalog)
            quantity = self.rng.randint(1, 5)
            price = round(self.rng.uniform(9.99, 399.99), 2)
            items.append({"product_name": product, "quantity": quantity, "price": price})

        if scenario == "repeated_items" and items:
            items.append(dict(items[0]))

        return items

    def _build_order_payload(self, order_idx: int, scenario: str) -> dict[str, Any]:
        items = self._build_items(scenario)
        total = round(sum(item["quantity"] * item["price"] for item in items), 2)

        if scenario == "incorrect_total":
            delta = self.rng.choice([5.0, -7.5, 12.25])
            total = round(max(1.0, total + delta), 2)

        payload = {
            "order_id": f"ORD-{order_idx:07d}",
            "customer_name": self.rng.choice(self.customer_names),
            "items": items,
            "total_amount": float(total),
            "shipping_address": self.rng.choice(self.addresses),
            "order_date": self._random_order_date(),
        }

        if scenario == "missing_partial":
            payload["shipping_address"] = "UNKNOWN"

        return payload

    @staticmethod
    def _item_text(item: dict[str, Any]) -> str:
        return f"{item['quantity']} x {item['product_name']} @ ${item['price']:.2f}"

    def _build_input_text(self, payload: dict[str, Any], scenario: str, style: str) -> str:
        items_txt = ", ".join(self._item_text(item) for item in payload["items"])
        order_line = (
            f"Order {payload['order_id']} for {payload['customer_name']} includes {items_txt}. "
            f"Ship to {payload['shipping_address']} on {payload['order_date']}. "
            f"Total charged: ${payload['total_amount']:.2f}."
        )

        if style == "casual":
            order_line = (
                f"hey, it's {payload['customer_name']} ({payload['order_id']}). got {items_txt}; "
                f"send it to {payload['shipping_address']}. total came out to ${payload['total_amount']:.2f} "
                f"on {payload['order_date']}"
            )
        elif style == "shorthand":
            order_line = (
                f"id={payload['order_id']} name={payload['customer_name']} items:[{items_txt}] "
                f"addr={payload['shipping_address']} date={payload['order_date']} total={payload['total_amount']:.2f}"
            )

        if scenario == "ambiguous":
            order_line += " Note: maybe add gift wrap later, but current charge is what matters."
        elif scenario == "messy":
            order_line = f"***RAW*** {order_line} ::: copied from chat log!!!"
        elif scenario == "missing_partial":
            order_line += " Address line was not provided clearly in the source message."
        elif scenario == "incorrect_total":
            order_line += " Customer notes subtotal math looked odd; keep stated final total."
        elif scenario == "noisy":
            noise = self.rng.choice(["<html>n/a</html>", "#@$%", "[attachment missing]"])
            order_line = f"{noise} {order_line} {noise}"

        return order_line

    def _generate_single_synthetic_sample(self, order_idx: int) -> dict[str, str]:
        scenario = self.rng.choice(self.scenarios)
        style = self.rng.choice(self.styles)
        payload = self._build_order_payload(order_idx=order_idx, scenario=scenario)
        input_text = self._build_input_text(payload=payload, scenario=scenario, style=style)

        return {
            "instruction": self.prompt_templates.instruction,
            "input": input_text,
            "output": json.dumps(payload, ensure_ascii=False),
        }

    def generate_synthetic_dataset(self, n_samples: int, batch_size: int, max_retries: int) -> list[dict[str, str]]:
        samples: list[dict[str, str]] = []
        order_idx = 1

        while len(samples) < n_samples:
            remaining = n_samples - len(samples)
            current_batch_size = min(batch_size, remaining)
            LOGGER.info("Generating synthetic batch of %s samples (%s/%s)", current_batch_size, len(samples), n_samples)

            for _ in range(current_batch_size):
                accepted = False
                for _attempt in range(1, max_retries + 1):
                    candidate = self._generate_single_synthetic_sample(order_idx=order_idx)
                    validation = validate_alpaca_sample(candidate, self.schema)
                    if validation.is_valid:
                        samples.append(candidate)
                        accepted = True
                        order_idx += 1
                        break

                if not accepted:
                    LOGGER.warning("Discarded a synthetic sample after %s retries.", max_retries)

        return samples

    def generate_golden_dataset(self, n_samples: int) -> list[dict[str, str]]:
        if n_samples < 100 or n_samples > 300:
            raise ValueError("Golden dataset size must be between 100 and 300 samples.")

        scenarios = [
            "simple",
            "multi_item",
            "ambiguous",
            "messy",
            "missing_partial",
            "incorrect_total",
            "repeated_items",
            "noisy",
        ]

        golden_samples: list[dict[str, str]] = []
        base_date = date(2025, 1, 1)

        for idx in range(n_samples):
            scenario = scenarios[idx % len(scenarios)]
            style = self.styles[idx % len(self.styles)]

            items = [
                {
                    "product_name": self.product_catalog[(idx + item_idx) % len(self.product_catalog)],
                    "quantity": (idx + item_idx) % 4 + 1,
                    "price": round(19.99 + ((idx + item_idx) % 9) * 7.5, 2),
                }
                for item_idx in range(1 if scenario == "simple" else 2)
            ]

            if scenario == "repeated_items":
                items.append(dict(items[0]))

            total = round(sum(item["quantity"] * item["price"] for item in items), 2)
            if scenario == "incorrect_total":
                total = round(total + 4.25, 2)

            payload = {
                "order_id": f"GLD-{idx + 1:05d}",
                "customer_name": self.customer_names[idx % len(self.customer_names)],
                "items": items,
                "total_amount": float(total),
                "shipping_address": (
                    "UNKNOWN"
                    if scenario == "missing_partial"
                    else self.addresses[idx % len(self.addresses)]
                ),
                "order_date": (base_date + timedelta(days=idx)).isoformat(),
            }

            sample = {
                "instruction": self.prompt_templates.instruction,
                "input": self._build_input_text(payload=payload, scenario=scenario, style=style),
                "output": json.dumps(payload, ensure_ascii=False),
            }

            validation = validate_alpaca_sample(sample, self.schema)
            if not validation.is_valid:
                raise ValueError(f"Deterministic golden sample invalid at index {idx}: {validation.errors}")

            golden_samples.append(sample)

        return golden_samples


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
