<#
.SYNOPSIS
  AI Party — start a worker node (client.py).

.DESCRIPTION
  Registers this machine with an API host, spawns ggml-rpc-server, and
  starts heartbeating. The host will include this machine in the next
  tensor split.

.PARAMETER ApiUrl
  Base URL of the API host, e.g. http://192.168.1.6:8000.
  Required.

.PARAMETER NodeIp
  This machine's IP as seen by the API host. Auto-detected if not given.

.PARAMETER LlamaDir
  Folder containing ggml-rpc-server. Defaults to <repo>\bin.

.EXAMPLE
  .\scripts\run-worker.ps1 -ApiUrl http://192.168.1.6:8000

.EXAMPLE
  .\scripts\run-worker.ps1 -ApiUrl http://192.168.1.6:8000 -NodeIp 192.168.1.14

.EXAMPLE
  .\scripts\run-worker.ps1 -ApiUrl http://192.168.1.6:8000 -Port 50053
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)] [string]$ApiUrl,
  [string]$NodeIp = "",
  [string]$LlamaDir = "",
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $LlamaDir) { $LlamaDir = Join-Path $RepoRoot "bin" }

# ---------- auto-detect node IP if not given ----------
if (-not $NodeIp) {
  Write-Host "No -NodeIp provided, auto-detecting..." -ForegroundColor Yellow
  $ip = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object {
          $_.IPAddress -notlike "127.*" -and
          $_.IPAddress -notlike "169.254.*" -and
          $_.PrefixOrigin -ne "WellKnown"
        } |
        Sort-Object -Property SkipAsSource |
        Select-Object -First 1 -ExpandProperty IPAddress

  if (-not $ip) {
    Write-Host ""
    Write-Host "Could not auto-detect a local IP." -ForegroundColor Red
    Write-Host "Pass -NodeIp explicitly. This must be the address the API host" -ForegroundColor Red
    Write-Host "uses to reach THIS machine, not the other way around." -ForegroundColor Red
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Red
    Write-Host "  -NodeIp 192.168.1.14      (LAN)"
    Write-Host "  -NodeIp 100.99.40.83      (Tailscale)"
    Write-Host "  -NodeIp 10.0.0.5          (VPN)"
    exit 1
  }
  $NodeIp = $ip
  Write-Host "Auto-detected node IP: $NodeIp" -ForegroundColor Green
}

# ---------- pick Python ----------
$py = $null
foreach ($candidate in @(
  (Join-Path $RepoRoot "myvenv\Scripts\python.exe"),
  (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
  (Join-Path $RepoRoot "venv\Scripts\python.exe")
)) {
  if (Test-Path $candidate) {
    $py = $candidate
    break
  }
}

if (-not $py) {
  $pyCmd = Get-Command python -ErrorAction SilentlyContinue
  if (-not $pyCmd) {
    Write-Host "Python not found. Run .\scripts\setup.ps1 first." -ForegroundColor Red
    exit 1
  }
  $py = $pyCmd.Source
}

# ---------- hand off to client.py ----------
Write-Host ""
Write-Host "Launching worker..." -ForegroundColor Yellow
Write-Host "  API host  : $ApiUrl"
Write-Host "  Node IP   : $NodeIp"
Write-Host "  llama dir : $LlamaDir"
Write-Host ""

& $py "$RepoRoot\client.py" `
  --api-url $ApiUrl `
  --node-ip $NodeIp `
  --llama-dir $LlamaDir `
  @ExtraArgs
