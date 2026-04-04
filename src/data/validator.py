from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from jsonschema import ValidationError, validate


@dataclass
class ValidationResult:
    json_valid: bool
    schema_valid: bool
    required_complete: bool
    types_valid: bool
    errors: list[str]

    @property
    def is_valid(self) -> bool:
        return self.json_valid and self.schema_valid and self.required_complete and self.types_valid


def parse_json_output(output: str) -> tuple[bool, dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(output)
        if not isinstance(payload, dict):
            return False, None, "parsed output is not a JSON object"
        return True, payload, None
    except json.JSONDecodeError as exc:
        return False, None, f"json_decode_error: {exc}"


def validate_schema(payload: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        validate(instance=payload, schema=schema)
        return True, None
    except ValidationError as exc:
        return False, f"schema_validation_error: {exc.message}"


def check_required_completeness(payload: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    required_fields = schema.get("required", [])
    for field in required_fields:
        if field not in payload or payload[field] in (None, "", []):
            missing.append(field)
    return len(missing) == 0, missing


def check_field_types(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    type_errors: list[str] = []

    if not isinstance(payload.get("order_id"), str):
        type_errors.append("order_id must be a string")
    if not isinstance(payload.get("customer_name"), str):
        type_errors.append("customer_name must be a string")
    if not isinstance(payload.get("shipping_address"), str):
        type_errors.append("shipping_address must be a string")
    if not isinstance(payload.get("order_date"), str):
        type_errors.append("order_date must be a string")
    if not isinstance(payload.get("total_amount"), (int, float)):
        type_errors.append("total_amount must be numeric")

    items = payload.get("items")
    if not isinstance(items, list):
        type_errors.append("items must be a list")
    else:
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                type_errors.append(f"items[{idx}] must be an object")
                continue
            if not isinstance(item.get("product_name"), str):
                type_errors.append(f"items[{idx}].product_name must be a string")
            if not isinstance(item.get("quantity"), int):
                type_errors.append(f"items[{idx}].quantity must be an integer")
            if not isinstance(item.get("price"), (int, float)):
                type_errors.append(f"items[{idx}].price must be numeric")

    return len(type_errors) == 0, type_errors


def validate_alpaca_sample(sample: dict[str, Any], schema: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []

    if "output" not in sample or not isinstance(sample["output"], str):
        return ValidationResult(
            json_valid=False,
            schema_valid=False,
            required_complete=False,
            types_valid=False,
            errors=["sample output missing or not string"],
        )

    json_valid, payload, parse_err = parse_json_output(sample["output"])
    if not json_valid or payload is None:
        errors.append(parse_err or "json_parse_failed")
        return ValidationResult(
            json_valid=False,
            schema_valid=False,
            required_complete=False,
            types_valid=False,
            errors=errors,
        )

    schema_valid, schema_err = validate_schema(payload, schema)
    if not schema_valid and schema_err:
        errors.append(schema_err)

    required_complete, missing = check_required_completeness(payload, schema)
    if not required_complete:
        errors.append(f"required_fields_missing_or_empty: {missing}")

    types_valid, type_errors = check_field_types(payload)
    if not types_valid:
        errors.extend(type_errors)

    return ValidationResult(
        json_valid=True,
        schema_valid=schema_valid,
        required_complete=required_complete,
        types_valid=types_valid,
        errors=errors,
    )
