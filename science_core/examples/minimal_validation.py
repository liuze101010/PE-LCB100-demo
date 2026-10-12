from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "generated_code" / "md_analysis"
sys.path.insert(0, str(MODULE_DIR))

from polymer_design_validator import validate_polymer_design_candidate


valid = validate_polymer_design_candidate(
    {
        "candidate_id": "example_polyether",
        "polymer_family": "polyether",
        "polymer_smiles": "CCOCC(C)CCN",
        "chain_count": 4,
    }
)
invalid = validate_polymer_design_candidate(
    {
        "candidate_id": "disconnected_example",
        "polymer_family": "polymer",
        "polymer_smiles": "CC.CC",
        "chain_count": 4,
    }
)

assert valid["validator_status"] == "pass", valid
assert invalid["validator_status"] == "reject", invalid
print(json.dumps({"valid": valid["validator_status"], "invalid": invalid["validator_status"]}))
