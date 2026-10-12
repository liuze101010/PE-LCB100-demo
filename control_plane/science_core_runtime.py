from __future__ import annotations

import hashlib
import os
import sys
import zipfile
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = PACKAGE_ROOT / "deploy" / "science_core_bundle.zip"
RUNTIME = PACKAGE_ROOT / ".runtime" / "control_science_core"


def extract_bundle(archive: zipfile.ZipFile, destination_root: Path) -> None:
    """Validate archive paths and support nested Windows installation paths."""
    destination_root = destination_root.resolve()
    for member in archive.infolist():
        destination = (destination_root / member.filename).resolve()
        if destination_root not in destination.parents and destination != destination_root:
            raise RuntimeError("science-core archive contains an unsafe path")
        writable = destination
        if os.name == "nt" and not str(destination).startswith("\\\\?\\"):
            writable = Path("\\\\?\\" + str(destination))
        if member.is_dir():
            writable.mkdir(parents=True, exist_ok=True)
        else:
            writable.parent.mkdir(parents=True, exist_ok=True)
            writable.write_bytes(archive.read(member))


def analysis_path() -> Path:
    if not BUNDLE.is_file():
        raise RuntimeError("the versioned science-core bundle is missing")
    digest = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    marker = RUNTIME / ".bundle_sha256"
    if not marker.is_file() or marker.read_text(encoding="ascii").strip() != digest:
        if RUNTIME.exists():
            import shutil

            removable = RUNTIME.resolve()
            if PACKAGE_ROOT.resolve() not in removable.parents:
                raise RuntimeError("science-core runtime must stay within the installed package")
            if os.name == "nt" and not str(removable).startswith("\\\\?\\"):
                removable = Path("\\\\?\\" + str(removable))
            shutil.rmtree(removable)
        RUNTIME.mkdir(parents=True)
        with zipfile.ZipFile(BUNDLE) as archive:
            extract_bundle(archive, RUNTIME)
        marker.write_text(digest + "\n", encoding="ascii")
    path = RUNTIME / "science_core" / "generated_code" / "md_analysis"
    if not (path / "polymer_qc_rules.py").is_file():
        raise RuntimeError("science-core extraction is incomplete")
    # Audit modules load their project paths at import time. Supply the shipped
    # portable configuration while allowing the operator's explicit override.
    config = path / "configs" / "pelcb100.json"
    if not config.is_file():
        raise RuntimeError("science-core project configuration is missing")
    if not os.environ.get("PELCB_PROJECT_CONFIG") and not os.environ.get("LCB_PROJECT_CONFIG"):
        os.environ["PELCB_PROJECT_CONFIG"] = str(config)
    os.environ.setdefault("PELCB_WARN_AUDIT_OUTPUT_DIR", str(RUNTIME / "audit_results"))
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path
