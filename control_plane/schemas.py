from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Role = Literal["admin", "researcher", "reviewer"]


class LoginRequest(BaseModel):
    username: str = Field(min_length=2, max_length=80)
    password: str = Field(min_length=12, max_length=256)


class UserCreate(BaseModel):
    username: str = Field(pattern=r"^[A-Za-z0-9_.-]{2,80}$")
    password: str = Field(min_length=12, max_length=256)
    role: Role = "researcher"


class ClusterCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(default=22, ge=1, le=65535)
    remote_username: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    host_key_sha256: str = Field(pattern=r"^SHA256:[A-Za-z0-9+/=_-]{20,}$")
    partition: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    account: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    remote_root: str = Field(pattern=r"^/[A-Za-z0-9_./-]+$")
    gromacs_command: str = "gmx"
    emc_command: str = "emc"
    python_command: str = "python3"
    runner_command: str = "polymer-agent-remote"
    module_init: list[str] = Field(default_factory=list, max_length=12)
    max_cpus: int = Field(default=32, ge=1, le=512)
    max_gpus: int = Field(default=4, ge=0, le=64)
    max_walltime_hours: int = Field(default=168, ge=1, le=720)
    allowed_nodes: list[str] = Field(default_factory=list, max_length=128)
    environment_mode: Literal["module", "container"] = "module"
    container_runtime: Literal["apptainer", "singularity"] | None = None
    container_image: str | None = Field(default=None, pattern=r"^/[A-Za-z0-9_./+:-]+$")
    pipeline_command: str = "pelcb-pipeline-core"
    science_install_mode: Literal["existing", "managed"] = "existing"
    gromacs_acceleration: Literal["auto", "cpu", "cuda"] = "auto"
    upstream_terms_accepted: bool = False

    @field_validator("gromacs_command", "emc_command", "python_command", "runner_command", "pipeline_command")
    @classmethod
    def command_is_bounded(cls, value: str) -> str:
        import re

        if not re.fullmatch(r"[A-Za-z0-9_./+-]+", value):
            raise ValueError("commands must be a binary name or absolute path without shell operators")
        return value

    @field_validator("module_init")
    @classmethod
    def modules_are_bounded(cls, values: list[str]) -> list[str]:
        import re

        allowed = re.compile(r"^(module load [A-Za-z0-9_./+:-]+|source /[A-Za-z0-9_./+-]+)$")
        if any(not allowed.fullmatch(value) for value in values):
            raise ValueError("module_init accepts only 'module load NAME' or 'source /absolute/path'")
        return values

    @field_validator("container_image")
    @classmethod
    def container_configuration_is_complete(cls, value: str | None, info):
        if info.data.get("environment_mode") == "container" and not value:
            raise ValueError("container mode requires an absolute container_image")
        return value

    @model_validator(mode="after")
    def environment_configuration_is_consistent(self):
        if self.environment_mode == "container" and (not self.container_runtime or not self.container_image):
            raise ValueError("container mode requires container_runtime and container_image")
        if self.environment_mode == "module" and (self.container_runtime or self.container_image):
            raise ValueError("module mode cannot set container_runtime or container_image")
        if self.science_install_mode == "managed" and self.environment_mode != "module":
            raise ValueError("managed science installation is available only in module/host mode")
        if self.science_install_mode == "managed" and not self.upstream_terms_accepted:
            raise ValueError("managed science installation requires acceptance of upstream terms")
        return self


class ClusterProvisionRequest(BaseModel):
    confirmation: Literal["install"]
    overwrite: bool = False


class DesignCandidate(BaseModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    design_key: str | None = Field(default=None, max_length=100)
    branch_points: int = Field(ge=1, le=5)
    branch_carbons: int = Field(ge=1, le=40)
    branch_distribution_code: str = Field(pattern=r"^[A-Za-z0-9_-]{1,30}$")
    placement_code: str | None = Field(default=None, max_length=40)
    topology: str | None = Field(default=None, max_length=40)
    target_total_carbons: int = Field(default=100, ge=20, le=2000)
    backbone_repeat_units: int | None = Field(default=None, ge=1, le=1000)
    branch_positions: list[int] | None = Field(default=None, max_length=12)
    macro_smiles: str | None = Field(default=None, max_length=4000)
    structure_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    design_rationale: str = Field(default="", max_length=500)
    builder_spec: dict | None = None
    emc_ready: bool = False


class DesignPlanningProvenance(BaseModel):
    engine: str = Field(max_length=120)
    model: str = Field(max_length=120)
    response_id: str = Field(max_length=200)
    requested_count: int = Field(ge=1, le=100)
    catalog_size: int = Field(ge=1, le=1000)
    catalog_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    strategy_summary: str = Field(max_length=500)
    attempt: int = Field(ge=1, le=3)
    mode: Literal["new_design", "reproduce_pelcb100"] = "new_design"
    benchmark_manifest_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class ProjectSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    natural_language_goal: str = Field(default="", max_length=4000)
    polymer_family: str = "PE"
    architecture: str = "pe_lcb"
    force_field: str = "opls/2024/opls-aa"
    temperature_k: float = Field(default=400.0, gt=0, le=1000)
    pressure_atm: float = Field(default=1.0, gt=0, le=1000)
    chain_count: int = Field(default=150, ge=1, le=10000)
    initial_density_g_cm3: float = Field(default=0.85, gt=0, le=5)
    production_ns: int = Field(default=400, ge=1, le=400)
    rule_profile: str = "pelcb100_validated_v1"
    planning: DesignPlanningProvenance | None = None
    candidates: list[DesignCandidate] = Field(min_length=1, max_length=1000)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    spec: ProjectSpec


class DesignDraftRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=4000)


class PolymerSetupPreviewRequest(BaseModel):
    repeat_unit_smiles: str = Field(min_length=1, max_length=4000)
    candidate_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    polymer_family: str | None = Field(default=None, max_length=100)
    architecture: str | None = Field(default=None, max_length=100)
    builder_backend: str | None = Field(default=None, max_length=100)
    force_field: str | None = Field(default=None, max_length=200)
    target_total_carbons_per_chain: int | None = Field(default=None, ge=1, le=10000)
    chain_count: int | None = Field(default=None, ge=1, le=10000)

    @field_validator("repeat_unit_smiles")
    @classmethod
    def strip_required_structure(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("repeat_unit_smiles cannot be blank")
        return value

    @field_validator("candidate_id", "polymer_family", "architecture", "builder_backend", "force_field")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ModelKeyUpdate(BaseModel):
    openai_api_key: str | None = Field(default=None, max_length=4096)
    deepseek_api_key: str | None = Field(default=None, max_length=4096)
    dashscope_api_key: str | None = Field(default=None, max_length=4096)
    zhipuai_api_key: str | None = Field(default=None, max_length=4096)

    @field_validator("openai_api_key", "deepseek_api_key", "dashscope_api_key", "zhipuai_api_key")
    @classmethod
    def normalize_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def at_least_one_key(self):
        if not any([self.openai_api_key, self.deepseek_api_key, self.dashscope_api_key, self.zhipuai_api_key]):
            raise ValueError("at least one non-empty API key is required")
        return self


class HostKeyLookup(BaseModel):
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(default=22, ge=1, le=65535)


class PublicKeyInstall(BaseModel):
    password: str | None = Field(default=None, min_length=1, max_length=4096)


class RunCreate(BaseModel):
    cluster_id: str


class RunApproval(BaseModel):
    max_repair_attempts: int = Field(default=1, ge=0, le=3)
    max_additional_md_ns: int = Field(default=400, ge=0, le=400)
    max_cases: int = Field(default=100, ge=1, le=1000)
    cpus_per_task: int = Field(default=8, ge=1, le=512)
    gpus: int = Field(default=1, ge=0, le=64)
    walltime_hours: int = Field(default=168, ge=1, le=720)
    max_parallel_cases: int = Field(default=8, ge=1, le=1000)
    confirmation: Literal["approve"]


class ReviewResolution(BaseModel):
    decision: Literal["accept_review", "retry", "reject"]
    note: str = Field(min_length=3, max_length=4000)
