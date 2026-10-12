"""Canonical labels and generic names for LCB figures.

The data tables keep full ``case_id`` values as stable keys.  Plotting code
should use the compact generic labels from this module instead.
"""

import csv
import json
import re
from pathlib import Path


TOPOLOGY_CODES = {
    "star3": "S3",
    "htype": "H",
    "dbmid": "DBm",
    "dbfar": "DBf",
    "comb": "Comb",
    "pompom": "PP",
    "tri": "Tri",
    "acomb": "AComb",
    "q4comb": "Q4Cb",
    "q4pompom": "Q4PP",
    "q4clust": "Q4Cl",
    "q4grad": "Q4G",
    "q5comb": "Q5Cb",
    "q5pompom": "Q5PP",
    "q5clust": "Q5Cl",
    "q5grad": "Q5G",
}

BRANCH_COLORS = {
    "1": "#2f6f9f",
    "2": "#3f8f6b",
    "3": "#d97b2b",
    "4": "#7457a6",
    "5": "#b84a4a",
}

TOPOLOGY_LINESTYLES = {
    "star3": "-",
    "htype": "--",
    "dbmid": "-.",
    "dbfar": ":",
    "comb": "-",
    "pompom": "--",
    "tri": "-.",
    "q4comb": "-",
    "q4pompom": "--",
    "q4clust": "-.",
    "q4grad": ":",
}

BRANCH_LENGTH_MARKERS = {
    "3": "o",
    "5": "s",
    "8": "^",
    "12": "D",
    "16": "P",
    "20": "X",
}

POSITION_PATTERN_CODES = {
    "mid": ("mid", "Mid"),
    "near-end": ("end", "End"),
    "near-start": ("end", "End"),
    "far-end": ("end", "End"),
    "uniform": ("uni", "Uni"),
    "inner-uniform": ("uni", "Uni"),
    "two-ends": ("term", "Term"),
    "two-ends+mid": ("term", "Term"),
    "terminal-split": ("term", "Term"),
    "H-asymmetric": ("asym", "Asym"),
    "h-asymmetric": ("asym", "Asym"),
    "end-cluster": ("clus", "Clus"),
    "clustered": ("clus", "Clus"),
    "center-packed": ("clus", "Clus"),
    "start-packed": ("end", "End"),
    "end-packed": ("end", "End"),
    "inner-split": ("split", "Split"),
    "wide-center": ("wide", "Wide"),
    "gradient": ("grad", "Grad"),
    "left-gradient": ("grad", "Grad"),
    "right-gradient": ("grad", "Grad"),
    "three-point": ("tri", "Tri"),
    "dumbbell-far": ("db", "DB"),
    "mid-pair": ("db", "DB"),
}


def default_output_root():
    return Path(__file__).resolve().parent / "output"


def parse_case_id(case_id):
    text = str(case_id)
    linear_control = re.search(r"lcb_([a-z]+)_lin_(\d+)_c(\d+)", text)
    if linear_control:
        family = linear_control.group(1).upper()
        seq = int(linear_control.group(2))
        return {
            "case_prefix": f"{family.lower()}lin{seq:03d}",
            "case_seq": str(seq),
            "polymer_family": family,
            "topology_label": "linear",
            "branch_count": "0",
            "branch_length": "0",
            "branch_distribution_code": "none",
            "branch_distribution_label": "None",
            "chain_carbons": str(int(linear_control.group(3))),
        }
    generic = re.search(r"lcb_([a-z]+)_(\d+)_bp(\d+)_bc(\d+)_pd([a-z0-9]+)_c(\d+)", text)
    if generic:
        family = generic.group(1).upper()
        seq = int(generic.group(2))
        return {
            "case_prefix": f"{family.lower()}{seq:03d}",
            "case_seq": str(seq),
            "polymer_family": family,
            "topology_label": "",
            "branch_count": str(int(generic.group(3))),
            "branch_length": str(int(generic.group(4))),
            "branch_distribution_code": generic.group(5),
            "branch_distribution_label": generic.group(5).title(),
            "chain_carbons": str(int(generic.group(6))),
        }
    match = re.search(r"lcb(\d+)_([A-Za-z0-9]+).*?_b(\d+)_l(\d+)_c(\d+)", text)
    if not match:
        return {
            "case_prefix": text.split("_")[0],
            "case_seq": "",
            "polymer_family": "",
            "topology_label": "",
            "branch_count": "",
            "branch_length": "",
            "branch_distribution_code": "",
            "branch_distribution_label": "",
            "chain_carbons": "",
        }
    seq = int(match.group(1))
    return {
        "case_prefix": f"lcb{seq:02d}",
        "case_seq": str(seq),
        "polymer_family": "PP",
        "topology_label": match.group(2),
        "branch_count": str(int(match.group(3))),
        "branch_length": str(int(match.group(4))),
        "branch_distribution_code": "",
        "branch_distribution_label": "",
        "chain_carbons": str(int(match.group(5))),
    }


def short_prefix(case_prefix):
    match = re.search(r"lcb(\d+)", str(case_prefix))
    if match:
        return f"L{int(match.group(1)):02d}"
    return str(case_prefix)


def topology_code(topology):
    key = str(topology or "").strip()
    return TOPOLOGY_CODES.get(key, key if key else "LCB")


def distribution_code(pattern):
    key = str(pattern or "").strip().lower()
    if key in POSITION_PATTERN_CODES:
        return POSITION_PATTERN_CODES[key][0]
    if key.startswith("comb-uniform"):
        return "uni"
    if key.startswith("pompom"):
        return "term"
    if key.startswith("cluster"):
        return "clus"
    if key.startswith("gradient"):
        return "grad"
    return re.sub(r"[^a-z0-9]+", "", key) or "na"


def distribution_label(pattern_or_code):
    key = str(pattern_or_code or "").strip().lower()
    if key in POSITION_PATTERN_CODES:
        return POSITION_PATTERN_CODES[key][1]
    labels = {
        "mid": "Mid",
        "end": "End",
        "uni": "Uni",
        "term": "Term",
        "clus": "Clus",
        "grad": "Grad",
        "tri": "Tri",
        "db": "DB",
        "asym": "Asym",
        "split": "Split",
        "wide": "Wide",
        "na": "NA",
    }
    code = distribution_code(key) if key not in labels else key
    return labels.get(code, code.title())


def clean_int_text(value):
    text = str(value or "").strip()
    if text == "":
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def clean_seq(value, fallback_case_id=""):
    text = clean_int_text(value)
    if text:
        return text
    parsed = parse_case_id(fallback_case_id)
    return clean_int_text(parsed.get("case_seq", ""))


def polymer_family_from_row(row=None, case_id=""):
    if row is None:
        row = {}
    family = str(
        row.get("polymer_family")
        or row.get("repeat_unit")
        or row.get("backbone_monomer")
        or ""
    ).strip().lower()
    if family in {"pe", "polyethylene", "ethylene"}:
        return "PE"
    if family in {"pp", "ipp", "polypropylene", "propylene"}:
        return "PP"
    parsed = parse_case_id(case_id)
    return parsed.get("polymer_family") or "PP"


def generic_case_name_from_parts(polymer_family, seq, branch_count, branch_length, distribution, chain_carbons):
    family = str(polymer_family or "PP").strip().lower()
    seq_text = int(clean_int_text(seq) or 0)
    branch = int(clean_int_text(branch_count) or 0)
    length = int(clean_int_text(branch_length) or 0)
    carbons = int(clean_int_text(chain_carbons) or 0)
    dist = distribution_code(distribution)
    return f"lcb_{family}_{seq_text:03d}_bp{branch:02d}_bc{length:03d}_pd{dist}_c{carbons:03d}"


def generic_plot_label_from_parts(polymer_family, seq, branch_count, branch_length, distribution):
    family = str(polymer_family or "PP").strip().upper()
    seq_text = int(clean_int_text(seq) or 0)
    branch = clean_int_text(branch_count)
    length = clean_int_text(branch_length)
    dist = distribution_label(distribution)
    return f"{family}{seq_text:03d}/BP{branch}-BC{length}-{dist}"


def generic_metadata(case_id, row=None):
    if row is None:
        row = {}
    parsed = parse_case_id(case_id)
    seq = clean_seq(row.get("case_seq") or row.get("sequence"), case_id)
    family = polymer_family_from_row(row, case_id)
    branch_count = clean_int_text(
        row.get("branch_point_count")
        or row.get("branch_points")
        or row.get("branch_count")
        or parsed.get("branch_count")
    )
    branch_length = clean_int_text(
        row.get("branch_arm_carbons")
        or row.get("branch_carbons")
        or row.get("branch_length")
        or parsed.get("branch_length")
    )
    chain_carbons = clean_int_text(
        row.get("actual_total_carbons_per_chain")
        or row.get("chain_carbons")
        or parsed.get("chain_carbons")
    )
    pattern = (
        row.get("branch_distribution_code")
        or row.get("branch_position_pattern")
        or parsed.get("branch_distribution_code")
    )
    dist_code = distribution_code(pattern)
    dist_label = distribution_label(pattern)
    generic_case_name = generic_case_name_from_parts(
        family,
        seq,
        branch_count,
        branch_length,
        dist_code,
        chain_carbons,
    )
    generic_plot_label = generic_plot_label_from_parts(
        family,
        seq,
        branch_count,
        branch_length,
        dist_code,
    )
    legacy_topology = (
        row.get("legacy_template_topology")
        or row.get("branch_topology")
        or row.get("topology_label")
        or row.get("topology_slug")
        or parsed.get("topology_label")
    )
    return {
        "source_case_id": str(case_id),
        "generic_case_name": generic_case_name,
        "generic_plot_label": generic_plot_label,
        "polymer_family": family,
        "branch_point_count": branch_count,
        "branch_arm_carbons": branch_length,
        "branch_distribution_code": dist_code,
        "branch_distribution_label": dist_label,
        "legacy_template_topology": str(legacy_topology or ""),
        "naming_version": "generic_v1",
    }


def plot_label_from_parts(case_prefix, topology, branch_length, branch_count=""):
    length = clean_int_text(branch_length)
    branch = clean_int_text(branch_count)
    branch_part = f"B{branch}-" if branch else ""
    return f"{short_prefix(case_prefix)}/{branch_part}{topology_code(topology)}-L{length}"


def plot_label(case_id, row=None):
    if row and row.get("generic_plot_label"):
        return str(row["generic_plot_label"])
    if row:
        return generic_metadata(case_id, row)["generic_plot_label"]
    values = parse_case_id(case_id)
    if values.get("branch_count") and values.get("branch_length") and values.get("case_seq"):
        return generic_metadata(case_id, values)["generic_plot_label"]
    return plot_label_from_parts(values["case_prefix"], values["topology_label"], values["branch_length"], values["branch_count"])


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_case_label_rows(output_root=None):
    output_root = Path(output_root) if output_root else default_output_root()
    integrated = output_root / "lcb300_integrated" / "aggregate" / "lcb300_summary_wide.csv"
    if not integrated.exists():
        integrated = output_root / "lcb100_integrated" / "aggregate" / "structure_dynamics_summary.csv"
    rows = []
    seen = set()
    for row in read_csv_rows(integrated):
        case_id = row.get("case_id", "")
        if not case_id or case_id in seen:
            continue
        parsed = parse_case_id(case_id)
        case_prefix = row.get("case_prefix") or parsed["case_prefix"]
        topology = row.get("topology_label") or parsed["topology_label"]
        branch_count = row.get("branch_count") or parsed["branch_count"]
        branch_length = row.get("branch_length") or parsed["branch_length"]
        generic = generic_metadata(case_id, row)
        out = {
            "case_id": case_id,
            **generic,
            "case_prefix": case_prefix,
            "plot_label": generic["generic_plot_label"],
            "topology_label": topology,
            "branch_count": clean_int_text(branch_count),
            "branch_length": clean_int_text(branch_length),
            "backbone_carbons": clean_int_text(row.get("backbone_carbons", "")),
            "chain_carbons": clean_int_text(row.get("chain_carbons") or parsed["chain_carbons"]),
        }
        rows.append(out)
        seen.add(case_id)
    rows.sort(key=lambda r: r["case_id"])
    return rows


def write_case_label_map(output_root=None):
    output_root = Path(output_root) if output_root else default_output_root()
    rows = build_case_label_rows(output_root)
    agg = output_root / "lcb300_integrated" / "aggregate"
    fields = [
        "case_id",
        "source_case_id",
        "generic_case_name",
        "generic_plot_label",
        "polymer_family",
        "branch_point_count",
        "branch_arm_carbons",
        "branch_distribution_code",
        "branch_distribution_label",
        "legacy_template_topology",
        "naming_version",
        "case_prefix",
        "plot_label",
        "topology_label",
        "branch_count",
        "branch_length",
        "backbone_carbons",
        "chain_carbons",
    ]
    write_csv(agg / "case_label_map.csv", rows, fields)
    (agg / "case_label_map.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(agg / "generic_name_map.csv", rows, fields)
    (agg / "generic_name_map.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return rows


def load_case_label_map(output_root=None):
    output_root = Path(output_root) if output_root else default_output_root()
    path = output_root / "lcb300_integrated" / "aggregate" / "case_label_map.csv"
    if not path.exists():
        rows = write_case_label_map(output_root)
    else:
        rows = read_csv_rows(path)
    return {row["case_id"]: row.get("generic_plot_label") or row.get("plot_label") for row in rows if row.get("case_id")}


def load_case_metadata_map(output_root=None):
    output_root = Path(output_root) if output_root else default_output_root()
    path = output_root / "lcb300_integrated" / "aggregate" / "case_label_map.csv"
    if not path.exists():
        rows = write_case_label_map(output_root)
    else:
        rows = read_csv_rows(path)
    return {row["case_id"]: row for row in rows if row.get("case_id")}


def label_for_case(case_id, labels=None):
    labels = labels or load_case_label_map()
    return labels.get(str(case_id), plot_label(str(case_id)))


def metadata_for_case(case_id, metadata=None):
    metadata = metadata or load_case_metadata_map()
    case_id = str(case_id)
    return metadata.get(case_id, parse_case_id(case_id))


def branch_color_for_value(branch_count, fallback="#8c98a4"):
    branch = clean_int_text(branch_count)
    return BRANCH_COLORS.get(branch, fallback)


def branch_color_for_case(case_id, metadata=None, fallback="#8c98a4"):
    row = metadata_for_case(case_id, metadata)
    return branch_color_for_value(row.get("branch_count", ""), fallback=fallback)


def topology_linestyle_for_case(case_id, metadata=None):
    row = metadata_for_case(case_id, metadata)
    topology = str(row.get("topology_label", ""))
    return TOPOLOGY_LINESTYLES.get(topology, "-")


def branch_length_marker_for_value(branch_length, fallback="o"):
    length = clean_int_text(branch_length)
    return BRANCH_LENGTH_MARKERS.get(length, fallback)


def branch_length_marker_for_case(case_id, metadata=None):
    row = metadata_for_case(case_id, metadata)
    return branch_length_marker_for_value(row.get("branch_length", ""))
