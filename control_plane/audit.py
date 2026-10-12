from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any


FORBIDDEN_AUDIT_KEYS = {
    "primary_audit_diagnosis",
    "audit_diagnosis",
    "diagnosis_candidates",
    "repair_instruction",
    "recommended_action",
    "recommended_reentry_module",
    "repair_tool",
    "available_repair_tools",
    "repair_affordances",
    "route",
}


def sanitize_audit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove deterministic conclusions before independent model review."""

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key not in FORBIDDEN_AUDIT_KEYS}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    result = clean(deepcopy(payload))
    leaked = find_forbidden_keys(result)
    if leaked:
        raise ValueError("audit sanitizer retained forbidden fields: " + ", ".join(sorted(leaked)))
    return result


def find_forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_AUDIT_KEYS:
                found.add(key)
            found.update(find_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(find_forbidden_keys(item))
    return found


def aggregate_panel_votes(votes: list[dict[str, Any]]) -> dict[str, Any]:
    """Require majority agreement on both cause and route.

    Model self-reported confidence is preserved as metadata but never used to
    choose a winner or authorize execution.
    """
    valid = [
        vote
        for vote in votes
        if vote.get("cause") and vote.get("route") and vote.get("information_sufficiency") in {"sufficient", "insufficient"}
    ]
    if len(valid) < 2:
        return {"decision": "review", "reason": "fewer_than_two_valid_votes", "votes": votes}
    pairs = Counter((vote["cause"], vote["route"]) for vote in valid)
    (cause, route), count = pairs.most_common(1)[0]
    if count < 2:
        return {"decision": "review", "reason": "no_cause_route_majority", "votes": votes}
    supporters = [vote for vote in valid if (vote["cause"], vote["route"]) == (cause, route)]
    if any(vote["information_sufficiency"] != "sufficient" for vote in supporters):
        return {
            "decision": "review",
            "reason": "majority_reports_insufficient_information",
            "cause": cause,
            "route": route,
            "votes": votes,
        }
    return {
        "decision": "route",
        "cause": cause,
        "route": route,
        "agreement": f"{count}/{len(valid)}",
        "model_self_confidence": [vote.get("confidence") for vote in supporters],
        "votes": votes,
    }
