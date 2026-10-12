#!/usr/bin/env python3
"""Generic LCB supervisor entrypoint."""

from __future__ import annotations

from lcb300_supervisor import main


if __name__ == "__main__":
    raise SystemExit(main())
