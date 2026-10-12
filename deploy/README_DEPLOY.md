# PE-LCB Research Platform deployment

## 1. Control-plane host

Use a Linux host on the laboratory network with Docker Engine and the Compose
plugin. Copy `.env.example` to `.env`, replace every placeholder, and set
`POLYMER_AGENT_DOMAIN` to an internal DNS name with HTTPS.

```bash
docker compose build
docker compose up -d
docker compose exec api python -m polymer_agent_demo.scripts.doctor --stage md --json
```

The first login uses `POLYMER_AGENT_BOOTSTRAP_ADMIN`. Remove the bootstrap
password from `.env` after the administrator has created named accounts and
restart the API and worker containers.

## 2. Per-user first-run wizard

1. The researcher creates a cluster connection in the Web interface.
2. The platform generates an Ed25519 key and displays only its public half.
3. The researcher appends the public key to their cluster account's
   `~/.ssh/authorized_keys`.
4. An administrator checks the host, partition and remote root, then authorizes
   the connection.
5. The researcher selects either the managed user-space science runtime, a
   reviewed module stack, or an existing Apptainer/Singularity image and clicks
   **auto setup**.
6. Auto setup discovers the environment, builds a privacy-scanned scientific
   core archive, and uploads the versioned `polymer-agent-remote` runner,
   bounded `site-pelcb-pipeline` adapter, `pelcb-pipeline-core` facade and core
   modules into the configured remote root, then runs preflight.
7. In managed mode the platform downloads `emc-pypi==2025.8.21` (the retained
   paper environment containing EMC 9.4.4) and GROMACS 2024.4 from
   official upstream locations, verifies fixed SHA256 digests, compiles
   GROMACS, and installs wrappers below the user's remote
   root. Missing site-owned prerequisites are returned as an administrator
   action list. The connection becomes `ready` only when the pinned host key, Slurm,
   GROMACS, EMC, Python, science pipeline, runner and write permissions all pass.
8. The researcher saves the model API keys needed by the selected stages. Keys are
   encrypted at rest and only configured/not-configured status is returned.
9. Natural-language molecular design uses the planning-model key. Audit requires
   the registered panel keys. Actual submission additionally requires a `ready`
   cluster and the original execution checks.

Private SSH keys are encrypted before database storage. Neither private keys,
laboratory LLM keys nor absolute remote paths are returned by list endpoints.

## 3. Remote requirements

The cluster account must provide `sbatch`, `squeue`, `sacct`, `scancel` and
Python 3. Managed installation additionally requires CMake, GNU Make and a
C++17 compiler; CUDA mode requires a site-compatible CUDA toolkit and driver.
The installer never invokes `sudo`. Existing mode requires the selected module
stack or reviewed container to provide EMC, GROMACS and force-field data.

When a new polymer omits its construction method or force field, the science
core infers the supported architecture from its repeat-unit or branch
description and selects only registered adapter options. Before MD, the remote
worker runs the EMC build and a GROMACS `grompp` probe. A successful probe
records force-field assignment and topology verification. Missing atom types,
unsupported elements or incomplete topology stop the run and create a review
item. The built-in registry currently provides OPLS-AA; another force field
requires a reviewed adapter plugin and its parameter files. Successful atom
typing does not by itself establish physical accuracy. New materials retain a
force-field review flag until compatible external validation is available.

The Web wizard automatically installs the runner, site adapter, pipeline-core
facade and a versioned source bundle containing the portable validation, rule,
profile, audit and re-gating modules. The bundle is rebuilt before deployment;
an identical manifest is skipped, whereas changed source installs under a new
version. The managed installer obtains EMC/GROMACS directly from upstream and
records source URLs, versions, SHA256 values and acceleration mode in
`runtime_manifest.json`. Production remains non-ready until the staged worker
self-test confirms Python, EMC, GROMACS, Slurm, the science-core bundle and
write permissions on the selected cluster. Sites can still provide
`POLYMER_AGENT_PRODUCTION_PIPELINE` when they need a local wrapper, but the
default production contract is the built-in stage runner:

```bash
pelcb-pipeline-core self-test --json
polymer-agent-remote stage --stage build --project-spec project.json \
  --manifest manifest.json --work-root work
polymer-agent-remote stage --stage md --project-spec project.json \
  --manifest manifest.json --work-root work
polymer-agent-remote stage --stage postprocess --project-spec project.json \
  --manifest manifest.json --work-root work
```

For sites that disable automated upload, copy `deploy/remote_runner.py` and
`deploy/install_remote_runner.sh` to the cluster and install the runner against
the site's reviewed adapter manually:

```bash
bash install_remote_runner.sh /absolute/path/to/site-pelcb-pipeline
polymer-agent-remote self-test --json
```

The installed core reports `acceptance_ready`, `science_core_installed`,
`production_worker` and `production_ready` separately. The Web API cannot submit
arbitrary commands and never substitutes the minute smoke profile for a
production run. If a site blocks automated software installation, the same stage
contract can be backed by an approved module stack or container.

On success the reconciler writes a release snapshot with schema
`polymer_agent.remote_release.v1`, terminal `publication_state`, a 64-character
configuration hash, provenance records and case-level quality status,
property masks, audit votes and repair history. For the PE-LCB100 benchmark, the
snapshot is additionally compared with the frozen 92 PASS / 8 review oracle.
The oracle is an acceptance check, not a decision override: mismatches are
reported instead of silently rewritten. A case still carrying MSD-WARN at about
800 ns cannot be marked PASS. The reconciler collects only bounded JSON
artifacts and log summaries; large trajectories remain on the cluster.

Production submission remains blocked until:

- the cluster has status `ready`;
- the project manifest is validated and locked;
- the user confirms CPU/GPU/walltime and repair budgets;
- `POLYMER_AGENT_EXECUTOR_ENABLED=1`;
- Redis and the worker are healthy.

If `OPENAI_API_KEY` and `POLYMER_AGENT_UPPER_PLANNER=1` are configured, each
approved run records one hidden GPT upper-planning event before Slurm enqueue.
The returned plan is limited to registered execution and repair contracts; it
does not authorize PASS, change physical parameters or replace rule-layer
re-evaluation.

## 4. Scientific boundary

The validated automatic rule profile covers PE-LCB C100/C101, OPLS-AA, 400 K,
1 atm, 150 chains and the registered branch grammar. For a new melt polymer
with a valid chain or repeat-unit structure, the scientific core can search
RadonPy and build a target-specific density screening profile when
high-similarity records and compatible conditions are available. Medium,
distant or incomplete matches return `WARN: rule_profile_missing` and enter
the audit layer. A complete production run also requires a compatible
molecule-builder and simulation adapter. A clean trajectory may be extended
once from 400 ns to about 800 ns; persistent MSD warnings become final
review/non-PASS only after audit and bounded repair.
