param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [ValidateSet("replay")]
    [string]$Mode = "replay",
    [ValidateSet("auto", "demo")]
    [string]$Page = "auto",
    [switch]$NoBrowser,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
$snapshot = Join-Path $repoRoot "data\demo_snapshot.json"
Set-Location -LiteralPath $repoRoot

# Replay uses only the web dependencies. Select an interpreter installed for
# this checkout so an older, locally installed package cannot start instead.
$probe = @'
import importlib.util, pathlib, sys
missing = [name for name in ('uvicorn', 'fastapi', 'pydantic') if importlib.util.find_spec(name) is None]
if missing or sys.version_info < (3, 11):
    raise SystemExit(1)
spec = importlib.util.find_spec('polymer_agent_demo')
expected = pathlib.Path(sys.argv[1]).resolve() / '__init__.py'
raise SystemExit(0 if spec and spec.origin and pathlib.Path(spec.origin).resolve() == expected else 1)
'@
$candidates = @(
    @{ exe = $env:POLYMER_AGENT_PYTHON; argv = @() },
    @{ exe = (Join-Path $repoRoot ".venv\Scripts\python.exe"); argv = @() },
    @{ exe = (Join-Path $repoRoot ".venv-platform\Scripts\python.exe"); argv = @() },
    @{ exe = "python"; argv = @() },
    @{ exe = "py"; argv = @("-3") }
)
$picked = $null
foreach ($candidate in $candidates) {
    if (-not $candidate.exe) { continue }
    if ($candidate.exe -like "*.exe" -and -not (Test-Path -LiteralPath $candidate.exe)) { continue }
    try {
        $exe = $candidate.exe
        $argv = $candidate.argv
        & $exe @argv -c $probe $repoRoot 2>$null
        if ($LASTEXITCODE -eq 0) { $picked = $candidate; break }
    } catch { continue }
}
if (-not $picked) {
    throw @"
Replay needs Python 3.11+, FastAPI, uvicorn and Pydantic installed for this checkout.
From this directory, install the lightweight demo package:
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -e .
Then run .\start_demo.cmd again.
For an offline installation, use your local wheel directory with --no-index --find-links.
"@
}
if (-not (Test-Path -LiteralPath $snapshot -PathType Leaf)) {
    throw "Bundled snapshot is missing: data/demo_snapshot.json. Restore it from the release checkout."
}

$py = $picked.exe
$pyArgs = $picked.argv
$env:POLYMER_AGENT_APP_MODE = "demo"
$env:POLYMER_AGENT_DEMO_MODE = "replay"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
# The application blocks live features per service instance. Existing process
# credentials and project settings remain intact for explicit platform use.
& $py @pyArgs -m polymer_agent_demo.scripts.doctor --stage demo
if ($LASTEXITCODE -ne 0) { throw "Replay dependencies or bundled assets need attention; see doctor output." }

if (-not $NoBrowser) {
    Start-Job -ArgumentList $Port, $Page -ScriptBlock {
        param($p, $targetPage)
        for ($i = 0; $i -lt 60; $i++) {
            try {
                Invoke-WebRequest "http://127.0.0.1:$p/api/health" -UseBasicParsing -TimeoutSec 1 | Out-Null
                Start-Process "http://127.0.0.1:$p/$targetPage"
                break
            } catch { Start-Sleep -Milliseconds 500 }
        }
    } | Out-Null
}
Write-Host "Offline Polymer-Agent replay: http://127.0.0.1:$Port/$Page"
Write-Host "Interpreter: $py $($pyArgs -join ' ')"
& $py @pyArgs -m uvicorn polymer_agent_demo.app:app --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
