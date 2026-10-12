"""Strict schema for tri-LLM audit judgments."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


Verdict = Literal["pass", "warn", "fail"]


class Judgment(BaseModel):
    """Public JSON verdict returned by tri_llm_judge.judge()."""

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    agree_with_rule: bool | None
    rationale: str = Field(min_length=1)
    ambiguity_flag: str = ""


def validate_judgment(payload: dict[str, Any]) -> Judgment:
    """Validate a dictionary as a Judgment with a stable helper name."""

    try:
        return Judgment.model_validate(payload)
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        return Judgment.parse_obj(payload)


def judgment_schema_json() -> dict[str, Any]:
    """Return the JSON schema used in prompts."""

    try:
        return Judgment.model_json_schema()
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        return Judgment.schema()


__all__ = ["Judgment", "ValidationError", "Verdict", "judgment_schema_json", "validate_judgment"]
