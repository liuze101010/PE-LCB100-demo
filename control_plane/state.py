from __future__ import annotations


RUN_TRANSITIONS: dict[str, set[str]] = {
    "planned": {"awaiting_approval", "cancelled"},
    "awaiting_approval": {"queued", "cancelled"},
    "queued": {"running", "failed", "cancelled"},
    "running": {"collecting", "failed", "cancelled"},
    "collecting": {"postprocessing", "failed"},
    "postprocessing": {"evaluating", "failed"},
    "evaluating": {"repairing", "pass", "review", "failed"},
    "repairing": {"queued", "evaluating", "review", "failed", "cancelled"},
    "pass": set(),
    "review": {"queued", "cancelled"},
    "failed": set(),
    "cancelled": set(),
}


def require_transition(current: str, target: str) -> None:
    if target not in RUN_TRANSITIONS.get(current, set()):
        raise ValueError(f"invalid run transition: {current} -> {target}")
