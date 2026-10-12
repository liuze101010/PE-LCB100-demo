from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from polymer_agent_demo.snapshot import build_demo_snapshot


if __name__ == "__main__":
    output = SCRIPT.parents[1] / "data" / "demo_snapshot.json"
    snapshot = build_demo_snapshot(PROJECT_ROOT, output)
    print(
        f"Built {output.name}: "
        f"{len(snapshot['cases'])} cases, "
        f"{len(snapshot['source_manifest'])} hashed sources"
    )
