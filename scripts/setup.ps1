<#
.SYNOPSIS
  AI Party — setup script for Windows.

.DESCRIPTION
  Auto-detects architecture and NVIDIA GPU, then downloads the correct
  prebuilt llama.cpp binaries from GitHub.

  Installs the full dependency set. A machine can act as an API host,
  a worker node, or both — the role is chosen at runtime.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

# ---------- config ----------
$RepoRoot  = Split-Path -Parent $PSScriptRoot
$BinDir    = Join-Path $RepoRoot "bin"
$ModelsDir = Join-Path $RepoRoot "models"
$VenvDir   = Join-Path $RepoRoot ".venv"

$LlamaRepo = "ggml-org/llama.cpp"
$GitHubApi = "https://api.github.com/repos/$LlamaRepo/releases"
$GitHubDl  = "https://github.com/$LlamaRepo/releases/download"

# ---------- pretty output ----------
function Say  ($msg) { Write-Host "> $msg" -ForegroundColor Yellow }
function Ok   ($msg) { Write-Host "+ $msg" -ForegroundColor Green }
function Warn ($msg) { Write-Host "! $msg" -ForegroundColor Yellow }
function Die  ($msg) { Write-Host "x $msg" -ForegroundColor Red; exit 1 }

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
  Say "NVIDIA GPU detected, querying driver version..."
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
Say "Checking Python..."
$py = $null
$ver = $null
foreach ($cand in @("python", "py")) {
  $cmd = Get-Command $cand -ErrorAction SilentlyContinue
  if ($cmd) {
    $verOutput = & $cand --version 2>&1
    if ($verOutput -match 'Python (\d+)\.(\d+)') {
      $majorVer = [int]$Matches[1]
      $minorVer = [int]$Matches[2]
      if ($majorVer -ge 3 -and $minorVer -ge 10) {
        $py = $cand
        $ver = "$majorVer.$minorVer"
        break
      }
    }
  }
}
if (-not $py) {
  Die @"
Python 3.10+ not found. Install from https://www.python.org/downloads/
Tick "Add python.exe to PATH" and "tcl/tk and IDLE" during setup.
Then re-run this script.
"@
}
Ok "Python $ver"

# ---------- Fetch latest release tag (bNNNNN) ----------
Say "Resolving latest llama.cpp release..."
try {
  $releases = Invoke-RestMethod -Uri "$GitHubApi?per_page=1" -UseBasicParsing
  $Tag = $releases[0].tag_name
} catch {
  Die "Could not reach GitHub API: $_"
}
if (-not $Tag) { Die "Could not resolve latest release tag." }
Ok "Latest release: $Tag"

# ---------- Select asset ----------
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

Say "Downloading $assetName ..."
try {
  Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
} catch {
  Die "Download failed: $_"
}

Say "Extracting into $BinDir ..."
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
Expand-Archive -Path $zipPath -DestinationPath $tmpDir -Force

$extracted = Get-ChildItem -Path $tmpDir -Directory | Where-Object { $_.Name -like "llama-*" } | Select-Object -First 1
if ($extracted) {
  Copy-Item -Path (Join-Path $extracted.FullName "*") -Destination $BinDir -Recurse -Force
} else {
  Copy-Item -Path (Join-Path $tmpDir "*") -Destination $BinDir -Recurse -Force
}

if ($GpuTag) {
  $runtimeUrl = "$GitHubDl/$Tag/$runtimeAssetName"
  $runtimeZip = Join-Path $tmpDir $runtimeAssetName
  Say "Checking for CUDA runtime DLLs..."
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
Say "Installing Python dependencies..."
$reqPath = Join-Path $RepoRoot "requirements.txt"
if (-not (Test-Path $reqPath)) { Die "Missing $reqPath" }

if (-not (Test-Path $VenvDir)) {
  Say "Creating virtual environment at $VenvDir"
  & $py -m venv $VenvDir
}
$venvPy = Join-Path $VenvDir "Scripts\python.exe"
& $venvPy -m pip install --upgrade pip | Out-Null
& $venvPy -m pip install -r $reqPath
Ok "Dependencies installed"

# ---------- Models ----------
if (-not (Test-Path $ModelsDir)) {
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

# ---------- Frontend check ----------
$frontendIndex = Join-Path $RepoRoot "frontend\index.html"
if (Test-Path $frontendIndex) {
  Ok "Frontend present at $RepoRoot\frontend"
} else {
  Warn "No prebuilt frontend at $RepoRoot\frontend."
  Warn "The API will run, but the chat UI won't be served."
}

# ---------- Summary ----------
Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "  Binaries : $BinDir"
Write-Host "  Models   : $ModelsDir"
Write-Host "  Venv     : $VenvDir"
Write-Host "  Build    : windows/$ArchTag $(if ($GpuTag) { "($GpuTag)" })"
Write-Host ""
Write-Host "You can now run this machine as a host, a worker, or both."
Write-Host ""
Write-Host "  Start as host (serves the API + chat UI):"
Write-Host "    .\scripts\run-host.ps1"
Write-Host ""
Write-Host "  Start as worker (joins another machine's cluster):"
Write-Host "    .\scripts\run-worker.ps1 -ApiUrl http://<host-ip>:8000"
Write-Host ""
