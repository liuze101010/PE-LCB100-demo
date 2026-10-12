"""Small XVG readers shared by MD analysis scripts."""

from pathlib import Path

import pandas as pd


def read_xvg(path, column_names=None):
    """Read a numeric XVG-like file into a DataFrame.

    Header lines beginning with '@' or '#' are ignored. The first numeric
    column is usually time in ps. When column_names is omitted, generic
    names value_1, value_2, ... are assigned after time_ps.
    """
    path = Path(path)
    rows = []
    width = None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("@") or line.startswith("#"):
            continue
        parts = line.split()
        try:
            row = [float(part) for part in parts]
        except ValueError:
            continue
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError(f"Inconsistent numeric column count in {path}: {len(row)} != {width}")
        rows.append(row)

    if not rows:
        raise ValueError(f"No numeric rows found in {path}")

    if column_names is None:
        if width == 1:
            column_names = ["time_ps"]
        else:
            column_names = ["time_ps"] + [f"value_{idx}" for idx in range(1, width)]
    if len(column_names) != width:
        raise ValueError(f"column_names has {len(column_names)} entries for {width} columns in {path}")

    frame = pd.DataFrame(rows, columns=column_names)
    return frame.sort_values(column_names[0]).drop_duplicates(subset=column_names[0], keep="last").reset_index(drop=True)


def write_json(path, payload):
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

