from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE_ROOT / "science_core"
OUTPUT = PACKAGE_ROOT / "deploy" / "science_core_bundle.zip"
SKIP = {"__pycache__", ".pytest_cache", ".venv", ".runtime"}
FIXTURE_DIRECTORY = (
    "generated_code/emc_alpha_olefin_builder/output/lcb100_bridge_fixed_20260424/"
    "lcb01_star3_b1_l05_c101"
)
FIXTURE_PATHS = frozenset(
    f"{FIXTURE_DIRECTORY}/{name}"
    for name in ("lcb01_star3_b1_l05_c101.gro", "topol.top", "lcb01_star3_b1_l05_c101_polymer.itp")
)
FORBIDDEN = [
    re.compile(rb"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)\d{1,3}\.\d{1,3}\b"),
    re.compile(rb"/(?:home|public/home)/[A-Za-z0-9_.-]+/"),
    re.compile(rb"[A-Za-z]:\\(?:Users|research-vault|lz project)\\", re.IGNORECASE),
    re.compile(rb"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
]


def source_root() -> Path:
    """Build exclusively from readable source shipped in this repository."""
    required = SOURCE / "generated_code" / "md_analysis" / "mcp" / "pelcb_mcp_server.py"
    if not required.is_file():
        raise FileNotFoundError("The repository science_core source is missing")
    return SOURCE


def selected_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file()
         and not any(part in SKIP for part in path.relative_to(root).parts)
         and path != root / "manifest.json"),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def packaged_fixtures(root: Path, version: str) -> list[tuple[str, bytes]]:
    """Restore only wheel-excluded fixtures from the shipped, same-version ZIP."""
    provenance = json.loads((root / "UPSTREAM.json").read_text(encoding="utf-8"))
    upstream = {row["path"]: row for row in provenance["files"]}
    missing = [relative for relative in upstream if not (root / relative).is_file()]
    missing_source = sorted(set(missing) - FIXTURE_PATHS)
    if missing_source:
        raise FileNotFoundError("Readable science-core source is missing: " + ", ".join(missing_source))
    missing_fixtures = sorted(set(missing) & FIXTURE_PATHS)
    if not missing_fixtures:
        return []
    if not OUTPUT.is_file():
        raise FileNotFoundError("The same-version ZIP containing wheel fixtures is missing")
    restored = []
    with zipfile.ZipFile(OUTPUT) as archive:
        manifest = json.loads(archive.read("science_core/manifest.json"))
        if manifest.get("version") != version:
            raise ValueError("The fixture ZIP version differs from the readable source")
        indexed = {row["path"]: row for row in manifest["files"]}
        for relative in missing_fixtures:
            data = archive.read("science_core/" + relative)
            digest = hashlib.sha256(data).hexdigest()
            if digest != upstream[relative]["upstream_sha256"] or digest != indexed[relative]["sha256"]:
                raise ValueError("Packaged fixture does not match its upstream SHA256: " + relative)
            restored.append((relative, data))
    return restored


def build() -> dict:
    root = source_root()
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    fallback = packaged_fixtures(root, version)
    records = []
    contents = [(path.relative_to(root).as_posix(), path.read_bytes()) for path in selected_files(root)]
    contents.extend(fallback)
    contents.sort(key=lambda item: item[0])
    for relative, data in contents:
        if any(pattern.search(data) for pattern in FORBIDDEN):
            raise ValueError(f"Refusing private or credential-like content: {relative}")
        records.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    manifest = {"schema_version": "polymer_agent.science_core_bundle.v2",
                "version": version, "file_count": len(records), "files": records}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, data in contents:
            info = zipfile.ZipInfo(f"science_core/{relative}", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
        info = zipfile.ZipInfo("science_core/manifest.json", date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, json.dumps(manifest, ensure_ascii=False, indent=2).encode())
    temporary.replace(OUTPUT)
    return {"output": str(OUTPUT), "version": version, "file_count": len(records),
            "size": OUTPUT.stat().st_size, "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
            "packaged_fixture_count": len(fallback),
            "fixture_storage": "verified_same_version_zip" if fallback else "readable_source"}


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
