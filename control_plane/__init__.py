"""Deployable PE-LCB control plane.

The existing replay demo remains available at ``/demo``.  This package owns
multi-user projects, cluster credentials, durable run state and release data.
"""

def init_database() -> None:
    """Initialize storage when a platform entrypoint explicitly requests it."""
    from .db import init_database as initialize

    initialize()

__all__ = ["init_database"]
