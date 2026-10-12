# Scientific workflow source

This directory contains the original MIT-licensed PE-LCB100 workflow source,
with the `generated_code/` layout preserved. `UPSTREAM.json` records each
upstream source and SHA256; the deployment bundle records current hashes and
the version in `VERSION`.

The baseline is the 2026-08-10 code release (`2fdaf77`). The teacher review
package contains the same selected source files. Local project-path handling
and public configuration are adapted for installation in another directory.

Run the deterministic molecular-input example after installing the scientific
dependencies from the repository root:

```bash
python science_core/examples/minimal_validation.py
```

Expected output: `{"valid": "pass", "invalid": "reject"}`.

The original tool API is
`generated_code/md_analysis/mcp/pelcb_mcp_server.py`. Set
`PELCB_PROJECT_CONFIG` to your own project JSON before starting this stdio MCP
entry point. Public configurations use environment placeholders for remote
accounts and paths. Register new inputs and output directories before invoking
execution tools.

The recommended production examples are `configs/pelcb100.public.json` and
the default `configs/pelcb100.json`, below `generated_code/md_analysis/`.
Every registered production profile declares GROMACS 2024.4. Fill your SSH
account and host, remote paths, allowed nodes and resource limits, and set
`gmx_bin` to your GROMACS 2024.4 executable. The retained `template.local.json`
and `template.server.json` files are **legacy comparison references**. They
preserve the earlier configuration shape; use the recommended examples for
new production work.

The directory includes construction and topology conversion, the original MD
templates and MDP files, deterministic input/result checks, rules and public
density references, model-audit schemas and prompts, repair routes, re-gating
and release bookkeeping. Model providers, cluster access, EMC and GROMACS are
separate runtime dependencies.

The small coordinate/topology files below the retained `output/` fixture path
are acceptance-test inputs already shipped by the public demonstration. They
contain no trajectories or model-response records. Production results and
private credentials are written to task-specific runtime directories.
