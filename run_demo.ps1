#Requires -Version 5.1
<#
.SYNOPSIS
    Launch the Thursday supervisor dashboard from saved Stage 3 outputs.

.DESCRIPTION
    Checks that the project venv and saved demo artefacts exist, then starts
    Streamlit on dashboard/app.py and opens http://localhost:8501.

    Always launches this repository's dashboard. A previous copy of the same
    demo on the target port is stopped first. An unrelated listener is an
    error — the script will not silently open another Streamlit app.

    Does not run X2DFD inference and does not modify any saved outputs.
#>
[CmdletBinding()]
param(
    [int]$Port = 8501,
    [int]$ReadyTimeoutSec = 60
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Fail([string]$Message) {
    Write-Host ""
    Write-Host "DEMO LAUNCH FAILED" -ForegroundColor Red
    Write-Host $Message
    Write-Host ""
    Write-Host "This launcher only serves saved Stage 3 outputs. It does not download"
    Write-Host "weights or run inference. Restore the missing file, then try again."
    exit 1
}

function Test-DemoUrl([int]$DemoPort) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$DemoPort/_stcore/health" `
            -UseBasicParsing -TimeoutSec 3
        return ($response.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Get-ListeningPid([int]$DemoPort) {
    $connections = @(
        Get-NetTCPConnection -LocalPort $DemoPort -State Listen -ErrorAction SilentlyContinue
    )
    if ($connections.Count -eq 0) {
        return $null
    }
    return [int]$connections[0].OwningProcess
}

function Get-ProcessCommandLine([int]$ProcessId) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if (-not $proc) {
        return $null
    }
    return [string]$proc.CommandLine
}

function Test-OurDashboardCommand([string]$CommandLine) {
    if (-not $CommandLine) {
        return $false
    }
    return (
        ($CommandLine -match 'streamlit') -and
        ($CommandLine -match 'dashboard([/\\]app\.py|_app\.py)')
    )
}

function Stop-ProcessTree([int]$ProcessId) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if (-not $proc) {
        return
    }
    $children = @(
        Get-CimInstance Win32_Process -Filter "ParentProcessId = $ProcessId" -ErrorAction SilentlyContinue
    )
    foreach ($child in $children) {
        if ($child.CommandLine -match 'streamlit') {
            Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
    $parentId = [int]$proc.ParentProcessId
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    if ($parentId -gt 0) {
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $parentId" -ErrorAction SilentlyContinue
        if ($parent -and ($parent.CommandLine -match 'streamlit')) {
            Stop-Process -Id $parentId -Force -ErrorAction SilentlyContinue
        }
    }
}

function Clear-DemoPort([int]$DemoPort) {
    $listenPid = Get-ListeningPid -DemoPort $DemoPort
    if (-not $listenPid) {
        return
    }
    $commandLine = Get-ProcessCommandLine -ProcessId $listenPid
    if (-not (Test-OurDashboardCommand -CommandLine $commandLine)) {
        $shown = if ($commandLine) { $commandLine } else { "(command line unavailable)" }
        Write-Fail (
            "Port $DemoPort is already in use by another process (PID $listenPid).`n" +
            "  $shown`n`n" +
            "Stop that process, or rerun with -Port <free-port>. This launcher will not " +
            "reuse an unrelated Streamlit app."
        )
    }
    Write-Host "Stopping previous demo server on port $DemoPort (PID $listenPid)."
    Stop-ProcessTree -ProcessId $listenPid

    $deadline = (Get-Date).AddSeconds(20)
    while (Get-ListeningPid -DemoPort $DemoPort) {
        if ((Get-Date) -gt $deadline) {
            Write-Fail "Port $DemoPort did not become free after stopping the previous demo server."
        }
        Start-Sleep -Milliseconds 400
    }
}

$RepoRoot = $PSScriptRoot
if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}
Set-Location -LiteralPath $RepoRoot

Write-Host "Repository root: $RepoRoot"

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$DashboardApp = Join-Path $RepoRoot "dashboard\app.py"
$Image = Join-Path $RepoRoot "datasets\raw\images\poc\real_face_01_crop.jpg"
$Summary = Join-Path $RepoRoot "eval\outputs\expert_matrix_summary.json"
$MatrixDir = Join-Path $RepoRoot "eval\outputs\expert_matrix\demo_one_crop"
$RequiredRuns = @(
    "demo_none.json",
    "demo_blending.json",
    "demo_diffusion.json",
    "demo_blending_diffusion.json"
)

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Fail "Missing virtualenv interpreter:`n  $Python`nCreate it first (see docs/RUNTIME_SETUP.md)."
}
if (-not (Test-Path -LiteralPath $DashboardApp)) {
    Write-Fail "Missing dashboard entry point:`n  $DashboardApp"
}
if (-not (Test-Path -LiteralPath $Image)) {
    Write-Fail "Missing demo image:`n  $Image"
}
if (-not (Test-Path -LiteralPath $Summary)) {
    Write-Fail "Missing Stage 3 summary:`n  $Summary"
}
if (-not (Test-Path -LiteralPath $MatrixDir)) {
    Write-Fail "Missing Stage 3 matrix directory:`n  $MatrixDir"
}
foreach ($name in $RequiredRuns) {
    $path = Join-Path $MatrixDir $name
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Fail "Missing Stage 3 result file:`n  $path"
    }
}

Write-Host "Checks passed:"
Write-Host "  python     $Python"
Write-Host "  image      $Image"
Write-Host "  summary    $Summary"
Write-Host "  matrix     $MatrixDir"
Write-Host ""
Write-Host "Launching Streamlit dashboard (saved outputs only; no inference)."

Clear-DemoPort -DemoPort $Port

$DashboardUrl = "http://localhost:$Port"
$streamlitArgs = @(
    "-m", "streamlit", "run", "dashboard/app.py",
    "--server.port", "$Port",
    "--server.address", "localhost",
    "--server.headless", "true",
    "--browser.gatherUsageStats", "false"
)
Start-Process -FilePath $Python -ArgumentList $streamlitArgs -WorkingDirectory $RepoRoot | Out-Null

$deadline = (Get-Date).AddSeconds($ReadyTimeoutSec)
while (-not (Test-DemoUrl -DemoPort $Port)) {
    if ((Get-Date) -gt $deadline) {
        Write-Fail "Streamlit did not become ready at $DashboardUrl within ${ReadyTimeoutSec}s."
    }
    Start-Sleep -Seconds 1
}

$listenPid = Get-ListeningPid -DemoPort $Port
$runningCmd = if ($listenPid) { Get-ProcessCommandLine -ProcessId $listenPid } else { "" }
if (-not (Test-OurDashboardCommand -CommandLine $runningCmd)) {
    Write-Fail (
        "Port $Port became ready, but the listener is not this dashboard.`n" +
        "  $runningCmd"
    )
}

Start-Process $DashboardUrl | Out-Null
Write-Host ""
Write-Host "Dashboard ready: $DashboardUrl" -ForegroundColor Green
Write-Host "Close the Streamlit window when the demo is finished."
exit 0
