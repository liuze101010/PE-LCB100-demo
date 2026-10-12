# Demo QA Report

## Deployable control-plane checks

- Multi-user sessions use HttpOnly SameSite cookies plus CSRF tokens.
- Project and cluster APIs enforce owner isolation; administrator access is explicit.
- Cluster list responses expose generated public keys but never encrypted private keys.
- PE-LCB projects are marked `validated_domain` only for the registered C100/C101 OPLS-AA protocol; other conditions create review records.
- Run state cannot skip approval, and production execution is disabled unless PostgreSQL, secure cookies and explicit deployment keys are configured.
- SSH host keys are pinned, remote paths are constrained under the configured root, and Slurm scripts use fixed arguments.
- Independent audit payloads remove deterministic diagnosis, candidate and repair-route fields before model calls.
- Cause and route must both reach a valid majority; model self-reported confidence does not select the winner.

## Functional Checks

- Three representative cases expose design, WARN source, model votes, route, repair evidence, and re-gate state.
- Route modules are MD for PE006, postprocess for PE002, and MD recovery for PE068.
- Route selection is explicitly two-stage: the first call returns the allowed routes, and the second accepts a guarded planning proposal.
- Connected and replay modes use the same public route schema.
- `execute=true` is rejected with HTTP 403.
- Model clients are preserved as recorded: `deepseek-v4-pro`, `qwen3-max`, and `glm-4.6`.
- Offline intent parsing preserves common requested families such as PEO, PVC, PTFE, PS, PA6, PPS, PDMS and PP instead of relabeling them as PE.
- Cross-material requests are visibly labeled as PE-LCB100 workflow replay, not material-specific evidence.
- Linear repeat units require exactly two `*` connection points; incomplete repeat-unit inputs are routed to review.

## Scientific Label Checks

- PE006 is described as pressure-relaxation/effective-run-parameter sensitivity, not a GROMACS-version effect.
- PE002 is described as a fit-window repair and does not imply an MD rerun.
- PE068 is recovered through the MD route and reaches the current strict PASS terminal state.
- Final cohort labels report the paper release state as `92 PASS / 8 review`; legacy ML-usable wording is not used as the main endpoint.
- The public audit ledger contains exactly 97 unique cases, 291 model votes, and 97 complete three-model panels.

## Visual Checks

- `screenshots/desktop.png`: 1600 x 1200 desktop view.
- `screenshots/mobile.png`: 500 x 2200 narrow responsive view.
- No incoherent overlaps were observed.
- Long case labels wrap on the narrow layout.
- Workflow, model votes, tool parameters, guard state, and before/after evidence remain readable.

## Security Checks

- Frozen snapshot exports logical source IDs and hashes instead of private source paths.
- No API keys, server usernames, remote home paths, or private absolute paths are exported.
- Runtime logs and browser profiles are excluded from the ZIP package.
- The current release checks are recorded in [VALIDATION.md](VALIDATION.md); the public tree passes `scripts/security_scan.py`.
