#!/usr/bin/env python3
"""Generic LCB EMC/bridge orchestrator.

This is the project-config driven entrypoint.  The LCB300-specific script is
kept for backward compatibility and shares the same implementation.
"""

from __future__ import annotations

from lcb300_emc_orchestrator import main


if __name__ == "__main__":
    raise SystemExit(main())
