# Tri-LLM Client Setup

This module supports three real API clients for the PE-LCB100 audit layer:

- DeepSeek V3+ family through the `deepseek-chat` endpoint
- Anthropic Claude Sonnet 4.6 through `claude-sonnet-4-6`
- Zhipu GLM through `glm-4.6`

The audit ledger records the paper-facing family labels
`deepseek-v3-plus`, `claude-sonnet-4-6`, and `glm-4.6`. Provider-reported
routed model ids should also be stored per call.

## Install Python SDKs

```bash
pip install "openai>=1.0" "anthropic>=0.40"
```

DeepSeek and GLM both use OpenAI-compatible endpoints through the `openai`
Python SDK.

## API Keys

Register and create API keys:

- DeepSeek platform: https://platform.deepseek.com/
- Anthropic Console: https://console.anthropic.com/
- ZhipuAI platform: https://open.bigmodel.cn/

PowerShell:

```powershell
$env:DEEPSEEK_API_KEY = "..."
$env:ANTHROPIC_API_KEY = "..."
$env:ZHIPUAI_API_KEY = "..."
```

Bash:

```bash
export DEEPSEEK_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export ZHIPUAI_API_KEY="..."
```

Optional model-id overrides:

```powershell
$env:DEEPSEEK_MODEL = "deepseek-chat"
$env:ANTHROPIC_MODEL = "claude-sonnet-4-6"
$env:GLM_MODEL = "glm-4.6"
```

Use overrides only when the provider API rejects the default model id. Record
the provider-reported model id in the SI-I frozen ledger.

## Known Provider Quirks

- DeepSeek uses an OpenAI-compatible endpoint at `https://api.deepseek.com`.
  The configured model id is `deepseek-chat`; production routing may report a
  newer backend.
- Claude Sonnet 4.6 does not provide native JSON mode in the Messages API.
  The prompt requests raw JSON only, but Claude can still return a fenced JSON
  block. `parse_judgment_payload()` handles this.
- GLM uses an OpenAI-compatible endpoint at
  `https://open.bigmodel.cn/api/paas/v4`. The locked paper configuration uses
  `glm-4.6` because it returned stable schema-valid JSON in A3 smoke tests.

## Cost Estimate

Measured PE-LCB100 audit calls were approximately:

- DeepSeek: about `550` input + `180` output tokens per call. At roughly
  `CNY 0.5 / 1M` input and `CNY 2 / 1M` output, the paper-scale DeepSeek pass is
  below `CNY 1`.
- Claude Sonnet 4.6: about `560` input + `240` output tokens per call. At
  roughly `$3 / 1M` input and `$15 / 1M` output, the paper-scale Claude pass is
  about `$1.63`.
- GLM 4.6: use the account balance shown in the ZhipuAI console; the PE-LCB100
  A3 smoke and benchmark are short strict-JSON calls, not long-horizon tasks.

The Level 3 smoke performs:

```text
3 decision types x 2 modes x 3 providers = 18 calls
```

## Run Level 3 Smoke

Default case:

```powershell
python generated_code\md_analysis\agentic\level3_smoke.py
```

Explicit case:

```powershell
python generated_code\md_analysis\agentic\level3_smoke.py `
  --case-id lcb_pe_001_bp01_bc005_pdmid_c101
```

Explicit evidence directory:

```powershell
python generated_code\md_analysis\agentic\level3_smoke.py `
  --case-id lcb_pe_001_bp01_bc005_pdmid_c101 `
  --case-dir generated_code\md_analysis\output\pelcb100_density_energy\lcb_pe_001_bp01_bc005_pdmid_c101
```

Local mock smoke without API calls:

```powershell
python generated_code\md_analysis\agentic\level3_smoke.py --mock
```

Force a new API run instead of using cache:

```powershell
python generated_code\md_analysis\agentic\level3_smoke.py --force-refresh
```

Outputs are written to:

```text
generated_code/md_analysis/output/pelcb100_supervisor/a1_level3_smoke_<timestamp>/
```

Expected files:

- `equilibration_blind.json`
- `equilibration_audit.json`
- `fit_quality_blind.json`
- `fit_quality_audit.json`
- `anomaly_blind.json`
- `anomaly_audit.json`
- `summary.csv`
- `smoke_health_report.md`
- `cache/*.json`

## Unit-Level Dry Run

```powershell
$env:TRI_LLM_USE_STATIC_CLIENTS = "1"
python -m pytest generated_code\md_analysis\agentic\tests\test_tri_llm_judge.py -q
```

Do not use static clients for paper metrics.
