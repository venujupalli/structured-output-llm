from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalMetrics:
    total: int = 0
    json_parse_success: int = 0
    schema_valid: int = 0
    required_complete: int = 0

    def to_dict(self) -> dict:
        total = max(self.total, 1)
        return {
            "total_samples": self.total,
            "json_parse_success_rate": self.json_parse_success / total,
            "schema_valid_rate": self.schema_valid / total,
            "required_field_completeness_rate": self.required_complete / total,
        }
