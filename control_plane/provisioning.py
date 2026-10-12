from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from .slurm import SSHSlurmBackend
from ..deploy.install_science_runtime import (
    EMC_DISTRIBUTIONS,
    GROMACS_SHA256,
    GROMACS_URL,
    GROMACS_VERSION,
    download,
)
from ..scripts.build_science_core_bundle import build as build_science_core_bundle


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_ROOT = PACKAGE_ROOT / "deploy"


def science_archive_cache() -> dict[str, Path]:
    cache_root = PACKAGE_ROOT / ".runtime" / "science_archive_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    gromacs_archive = cache_root / f"gromacs-{GROMACS_VERSION}.tar.gz"
    download(GROMACS_URL, gromacs_archive, GROMACS_SHA256)
    archives = {gromacs_archive.name: gromacs_archive}
    for distribution in EMC_DISTRIBUTIONS.values():
        emc_archive = cache_root / str(distribution["filename"])
        download(str(distribution["url"]), emc_archive, str(distribution["sha256"]))
        archives[emc_archive.name] = emc_archive
    return archives


def deployment_bundle() -> tuple[str, dict[str, bytes]]:
    build_science_core_bundle()
    paths = {
        "remote_runner.py": DEPLOY_ROOT / "remote_runner.py",
        "site_pipeline_adapter.py": DEPLOY_ROOT / "site_pipeline_adapter.py",
        "pelcb_pipeline_core.py": DEPLOY_ROOT / "pelcb_pipeline_core.py",
        "production_case.py": DEPLOY_ROOT / "production_case.py",
        "install_science_runtime.py": DEPLOY_ROOT / "install_science_runtime.py",
        "science_core_bundle.zip": DEPLOY_ROOT / "science_core_bundle.zip",
    }
    files = {name: path.read_bytes() for name, path in paths.items()}
    archive_members = {
        "run_real_smoke.py": "science_core/generated_code/lcb_md_pipeline/run_real_smoke.py",
        "pelcb_audit_diagnosis.py": "science_core/generated_code/md_analysis/pelcb_audit_diagnosis.py",
        "diagnosis_candidate_registry.py": "science_core/generated_code/md_analysis/diagnosis_candidate_registry.py",
        "mdp/steep.mdp": "science_core/generated_code/lcb_md_pipeline/mdp/steep.mdp",
        "fixture/lcb01_star3_b1_l05_c101.gro": "science_core/generated_code/emc_alpha_olefin_builder/output/lcb100_bridge_fixed_20260424/lcb01_star3_b1_l05_c101/lcb01_star3_b1_l05_c101.gro",
        "fixture/topol.top": "science_core/generated_code/emc_alpha_olefin_builder/output/lcb100_bridge_fixed_20260424/lcb01_star3_b1_l05_c101/topol.top",
        "fixture/lcb01_star3_b1_l05_c101_polymer.itp": "science_core/generated_code/emc_alpha_olefin_builder/output/lcb100_bridge_fixed_20260424/lcb01_star3_b1_l05_c101/lcb01_star3_b1_l05_c101_polymer.itp",
    }
    with zipfile.ZipFile(DEPLOY_ROOT / "science_core_bundle.zip") as archive:
        files.update({name: archive.read(member) for name, member in archive_members.items()})
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(files[name])
    version = "v1-" + digest.hexdigest()[:12]
    return version, files


def provision_cluster(backend: SSHSlurmBackend, *, overwrite: bool = False) -> dict:
    discovery = backend.discover_environment()
    if not discovery.get("platform_installable"):
        return {
            "ok": False,
            "installed": False,
            "discovery": discovery,
            "reason": "remote account cannot host platform components",
        }
    version, files = deployment_bundle()
    installation = backend.install_platform_stack(files=files, version=version, overwrite=overwrite)
    cache_seed = None
    if backend.runtime.environment.get("science_install_mode") == "managed":
        cache_seed = backend.seed_science_cache(archives=science_archive_cache())
    science_install = backend.start_managed_science_install(install_root=installation["install_root"])
    return {
        "ok": True,
        "installed": bool(installation.get("installed")),
        "discovery": discovery,
        "installation": installation,
        "science_cache_seed": cache_seed,
        "science_install": science_install,
    }
