param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [switch]$NoBrowser,
    [switch]$SkipInstall,
    [switch]$InstallOnly,
    [switch]$ResetDb,
    [switch]$KeepExistingServer,
    [string]$AdminUser = "admin",
    [string]$AdminPassword = "polymer-agent-local"
)

$ErrorActionPreference = "Stop"

$scriptRoot = $PSScriptRoot
$repoRoot = $scriptRoot
$sourceRootCandidate = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
$projectRoot = if (Test-Path -LiteralPath (Join-Path $sourceRootCandidate "generated_code\md_analysis")) {
    $sourceRootCandidate
} else {
    $repoRoot
}
$requirements = Join-Path $scriptRoot "requirements.txt"
$venvRoot = Join-Path $scriptRoot ".venv-platform"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
$marker = Join-Path $venvRoot ".requirements.sha256"
$runtimeRoot = Join-Path $scriptRoot ".runtime"
$dbPath = Join-Path $runtimeRoot "one_click_control_plane.db"
$dbMarker = Join-Path $runtimeRoot "one_click_control_plane.marker"

Set-Location $repoRoot

function Find-Python {
    $candidates = @(
        @{ exe = "py"; args = @("-3") },
        @{ exe = "python"; args = @() }
    )
    foreach ($candidate in $candidates) {
        try {
            $exe = $candidate.Item("exe")
            $argv = @($candidate.Item("args"))
            & $exe @argv -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        } catch {
            continue
        }
    }
    throw "Python 3.11+ was not found. Please install Python and enable Add python.exe to PATH."
}

function Invoke-Python([hashtable]$Python, [string[]]$Arguments) {
    $exe = $Python.Item("exe")
    $argv = @($Python.Item("args"))
    & $exe @argv @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $exe $($argv -join ' ') $($Arguments -join ' ')"
    }
}

function Install-Dependencies {
    if (-not (Test-Path $venvPython)) {
        Write-Host "Creating local virtual environment: $venvRoot"
        $basePython = Find-Python
        Invoke-Python $basePython @("-m", "venv", $venvRoot)
    }

    $requirementsHash = "{0}:{1}" -f `
        (Get-FileHash $requirements -Algorithm SHA256).Hash, `
        (Get-FileHash (Join-Path $scriptRoot "pyproject.toml") -Algorithm SHA256).Hash
    $installedHash = if (Test-Path $marker) { (Get-Content $marker -Raw).Trim() } else { "" }
    if ($installedHash -eq $requirementsHash) {
        Write-Host "Dependencies are already installed."
        return
    }

    Write-Host "Installing Python dependencies. First run may take a few minutes..."
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
    & $venvPython -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "dependency installation failed" }
    & $venvPython -m pip install -e ".[platform,science,dev]"
    if ($LASTEXITCODE -ne 0) { throw "editable package installation failed" }
    Set-Content -Path $marker -Value $requirementsHash -Encoding ASCII
}

function Assert-PortAvailable([int]$TargetPort) {
    $portProbe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $TargetPort)
    $portProbe.ExclusiveAddressUse = $true
    try {
        $portProbe.Start()
    } catch {
        throw "Port $TargetPort is occupied. Choose a free port with -Port, or close the existing server yourself."
    } finally {
        $portProbe.Stop()
    }
}

if (-not $InstallOnly) {
    Assert-PortAvailable $Port
}

New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null

if ($ResetDb -and (Test-Path $dbPath)) {
    Write-Host "Resetting local control-plane database: $dbPath"
    Remove-Item -LiteralPath $dbPath -Force
    Remove-Item -LiteralPath $dbMarker -Force -ErrorAction SilentlyContinue
}

if (-not $SkipInstall) {
    Install-Dependencies
} elseif (-not (Test-Path $venvPython)) {
    throw "SkipInstall was requested, but the local virtual environment does not exist: $venvPython"
}

if ($InstallOnly) {
    Write-Host "Installation complete. Start later with:"
    Write-Host "  .\start_platform.cmd"
    exit 0
}

if ((Test-Path $dbPath) -and -not (Test-Path $dbMarker)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backup = Join-Path $runtimeRoot "control_plane.before_one_click_$stamp.db"
    Write-Host "Existing local database was not created by the one-click launcher."
    Write-Host "Backing it up and creating a clean local login database:"
    Write-Host "  $backup"
    Move-Item -LiteralPath $dbPath -Destination $backup -Force
}

if (Test-Path -LiteralPath (Join-Path $projectRoot "generated_code\md_analysis")) {
    $env:POLYMER_AGENT_ROOT = $projectRoot
} else {
    Remove-Item Env:POLYMER_AGENT_ROOT -ErrorAction SilentlyContinue
}
$env:POLYMER_AGENT_APP_MODE = "platform"
$env:POLYMER_AGENT_DEMO_MODE = "replay"
$dbUriPath = $dbPath.Replace("\", "/")
$env:POLYMER_AGENT_DATABASE_URL = "sqlite:///$dbUriPath"
$env:POLYMER_AGENT_BOOTSTRAP_ADMIN = $AdminUser
$env:POLYMER_AGENT_BOOTSTRAP_PASSWORD = $AdminPassword
$env:POLYMER_AGENT_BOOTSTRAP_RESET = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if (-not $NoBrowser) {
    Start-Job -ArgumentList $Port -ScriptBlock {
        param($p)
        for ($i = 0; $i -lt 90; $i++) {
            try {
                Invoke-WebRequest "http://127.0.0.1:$p/api/health" -UseBasicParsing -TimeoutSec 1 | Out-Null
                Start-Process "http://127.0.0.1:$p/auto"
                break
            } catch {
                Start-Sleep -Milliseconds 500
            }
        }
    } | Out-Null
}

Write-Host ""
Write-Host "PE-LCB Research Platform is starting..."
Write-Host "离线演示 (打开即用): http://127.0.0.1:$Port/auto"
Write-Host "完整平台向导:        http://127.0.0.1:$Port/"
Write-Host "Login:    $AdminUser"
Write-Host "Password: $AdminPassword"
Write-Host ""
Write-Host "Keep this window open while using the platform. Press Ctrl+C to stop."
Write-Host ""

if (-not (Test-Path $dbMarker)) {
    Set-Content -Path $dbMarker -Value "admin=$AdminUser`ncreated_by=start_platform.cmd`n" -Encoding ASCII
}

& $venvPython -m uvicorn polymer_agent_demo.app:app --host 127.0.0.1 --port $Port
