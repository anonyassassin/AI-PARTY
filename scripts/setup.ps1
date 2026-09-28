<#
.SYNOPSIS
  AI Party — setup script for Windows.
.DESCRIPTION
  Auto-detects architecture and NVIDIA GPU, then downloads the correct
  prebuilt llama.cpp binaries from GitHub.
.PARAMETER Worker
  Install only worker-node dependencies.
#>
[CmdletBinding()]
param(
  [switch]$Worker
)

$ErrorActionPreference = "Stop"

# ---------- config ----------
$RepoRoot  = Split-Path -Parent $PSScriptRoot
$BinDir    = Join-Path $RepoRoot "bin"
$ModelsDir = Join-Path $RepoRoot "models"
$VenvDir   = Join-Path $RepoRoot ".venv"

$LlamaRepo = "ggml-org/llama.cpp"
$GitHubApi = "https://api.github.com/repos/$LlamaRepo/releases"
$GitHubDl  = "https://github.com/$LlamaRepo/releases/download"

$Mode = if ($Worker) { "worker" } else { "host" }

# ---------- pretty output ----------
function Say  ($msg) { Write-Host "▸ $msg" -ForegroundColor Yellow }
function Ok   ($msg) { Write-Host "✓ $msg" -ForegroundColor Green }
function Warn ($msg) { Write-Host "! $msg" -ForegroundColor Yellow }
function Die  ($msg) { Write-Host "✗ $msg" -ForegroundColor Red; exit 1 }

# ---------- Architecture detection ----------
$ArchTag = if ([Environment]::Is64BitOperatingSystem) {
  if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "x64" }
} else {
  Die "32-bit Windows is not supported."
}

# ---------- CUDA detection ----------
$GpuTag = ""
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
  Say "NVIDIA GPU detected, querying driver version…"
  try {
    $driverVer = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
    if ($driverVer) {
      $major = [int]($driverVer.Split('.')[0])
      if ($major -ge 570) {
        $GpuTag = "cuda-12.8"
        Ok "CUDA 12.8 capable (driver $driverVer)"
      } elseif ($major -ge 550) {
        $GpuTag = "cuda-12.4"
        Ok "CUDA 12.4 capable (driver $driverVer)"
      } else {
        Warn "Driver $driverVer too old for prebuilt CUDA archives; using CPU build."
      }
    }
  } catch {
    Warn "Could not query driver version; using CPU build."
  }
} else {
  Say "No NVIDIA GPU detected; using CPU build."
}

Say "Detected: windows / $ArchTag $(if ($GpuTag) { "(GPU: $GpuTag)" } else { "" })"

# ---------- Python ----------
Say "Checking Python…"
$py = $null
foreach ($cand in @("python", "py")) {
  $cmd = Get-Command $cand -ErrorAction SilentlyContinue
  if ($cmd) {
    $ver = & $cand -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if ($LASTEXITCODE -eq 0 -and $ver) {
      $parts = $ver.Split(".")
      if ([int]$parts[0] -ge 3 -and [int]$parts[1] -ge 10) { $py = $cand; break }
    }
  }
}
if (-not $py) {
  Die @"
Python 3.10+ not found. Install from https://www.python.org/downloads/
Tick "Add python.exe to PATH" during setup.
"@
}
Ok "Python $ver"

# ---------- Fetch latest release tag (bNNNNN) ----------
# /releases/latest returns stable vX.Y.Z; /releases?per_page=1 returns the
# newest build including rolling bNNNNN pre-releases, which is what we want.
Say "Resolving latest llama.cpp release…"
try {
  $releases = Invoke-RestMethod -Uri "$GitHubApi?per_page=1" -UseBasicParsing
  $Tag = $releases[0].tag_name
} catch {
  Die "Could not reach GitHub API: $_"
}
if (-not $Tag) { Die "Could not resolve latest release tag." }
Ok "Latest release: $Tag"

# ---------- Select asset ----------
# Windows naming:
#   CPU:  llama-<TAG>-bin-win-cpu-<ARCH>.zip
#   CUDA: llama-<TAG>-bin-win-cuda-12.4-<ARCH>.zip
#         llama-<TAG>-bin-win-cuda-12.8-<ARCH>.zip
if ($GpuTag) {
  $assetName = "llama-${Tag}-bin-win-${GpuTag}-${ArchTag}.zip"
  $runtimeAssetName = "cudart-llama-bin-win-${GpuTag}-${ArchTag}.zip"
} else {
  $assetName = "llama-${Tag}-bin-win-cpu-${ArchTag}.zip"
}

Say "Selected asset: $assetName"

# ---------- Download and extract ----------
$tmpDir = Join-Path $env:TEMP "llama-setup-$(Get-Random)"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

$url = "$GitHubDl/$Tag/$assetName"
$zipPath = Join-Path $tmpDir $assetName

Say "Downloading $assetName …"
try {
  Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
} catch {
  Die "Download failed: $_"
}

Say "Extracting into $BinDir …"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
Expand-Archive -Path $zipPath -DestinationPath $tmpDir -Force

# Flatten any versioned subdirectory
$extracted = Get-ChildItem -Path $tmpDir -Directory | Where-Object { $_.Name -like "llama-*" } | Select-Object -First 1
if ($extracted) {
  Copy-Item -Path (Join-Path $extracted.FullName "*") -Destination $BinDir -Recurse -Force
} else {
  Copy-Item -Path (Join-Path $tmpDir "*") -Destination $BinDir -Recurse -Force
}

# Optional CUDA runtime DLLs (for users without CUDA Toolkit installed)
if ($GpuTag) {
  $runtimeUrl = "$GitHubDl/$Tag/$runtimeAssetName"
  $runtimeZip = Join-Path $tmpDir $runtimeAssetName
  Say "Checking for CUDA runtime DLLs…"
  try {
    Invoke-WebRequest -Uri $runtimeUrl -OutFile $runtimeZip -UseBasicParsing -ErrorAction Stop
    Expand-Archive -Path $runtimeZip -DestinationPath $BinDir -Force
    Ok "CUDA runtime DLLs installed"
  } catch {
    Say "No separate CUDA runtime archive for this release."
  }
}

Remove-Item -Path $tmpDir -Recurse -Force

# ---------- Verify ----------
$serverExe = Join-Path $BinDir "llama-server.exe"
if (-not (Test-Path $serverExe)) {
  Die "llama-server.exe not found in $BinDir."
}
$rpcExe = Join-Path $BinDir "ggml-rpc-server.exe"
if (Test-Path $rpcExe) {
  Ok "Found llama-server.exe and ggml-rpc-server.exe"
} else {
  Warn "ggml-rpc-server.exe not found; RPC backend may be missing."
}

# ---------- Python deps ----------
Say "Installing Python dependencies…"
$reqFile = if ($Worker) { "requirements-worker.txt" } else { "requirements.txt" }
$reqPath = Join-Path $RepoRoot $reqFile
if (-not (Test-Path $reqPath)) { Die "Missing $reqPath" }

if (-not (Test-Path $VenvDir)) {
  Say "Creating virtual environment…"
  & $py -m venv $VenvDir
}
$venvPy = Join-Path $VenvDir "Scripts\python.exe"
& $venvPy -m pip install --upgrade pip | Out-Null
& $venvPy -m pip install -r $reqPath
Ok "Dependencies installed"

# ---------- Models ----------
if ($Mode -eq "host" -and -not (Test-Path $ModelsDir)) {
  New-Item -ItemType Directory -Path $ModelsDir | Out-Null
  @"
Drop your .gguf files in this folder.
Recommended starters:
  - Qwen2.5-0.5B-Instruct Q4_K_M   (~350 MB)
  - Qwen2.5-1.5B-Instruct Q4_K_M   (~1 GB)
  - Llama-3.2-3B-Instruct Q4_K_M   (~2 GB)
"@ | Out-File -Encoding utf8 (Join-Path $ModelsDir "README.txt")
  Ok "Created $ModelsDir"
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "  Binaries : $BinDir"
Write-Host "  Models   : $ModelsDir"
Write-Host "  Venv     : $VenvDir"
Write-Host "  Build    : windows/$ArchTag $(if ($GpuTag) { "($GpuTag)" })"
Write-Host ""

if ($Mode -eq "host") {
  Write-Host "Next steps (host):"
  Write-Host "  1. Add a .gguf file to $ModelsDir"
  Write-Host "  2. Start: .\scripts\run-host.ps1"
} else {
  Write-Host "Next steps (worker):"
  Write-Host "  .\scripts\run-worker.ps1 -ApiUrl http://<host>:8000"
}
