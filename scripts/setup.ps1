<#
.SYNOPSIS
  AI Party - setup script for Windows.

.DESCRIPTION
  Auto-detects architecture and NVIDIA GPU, then downloads the correct
  prebuilt llama.cpp binaries from GitHub.

  Installs the full dependency set. A machine can act as an API host,
  a worker node, or both - the role is chosen at runtime.
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

$gpuSummary = ""
if ($GpuTag) { $gpuSummary = " (GPU: $GpuTag)" }
Say "Detected: windows / $ArchTag$gpuSummary"

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
  Die "Python 3.10+ not found. Install from https://www.python.org/downloads/ and tick 'Add python.exe to PATH' and 'tcl/tk and IDLE' during setup. Then re-run this script."
}
Ok "Python $ver"

# ---------- Resolve a usable release tag ----------
# llama.cpp publishes rolling bNNNNN pre-releases. Older ones can be
# unpublished or marked as drafts, in which case their assets 404. We
# walk the recent releases and pick the newest one that:
#   - has at least one downloadable asset
#   - has an asset matching our selected Windows CPU/CUDA variant
# This avoids the "404 Not Found" that happens when /releases/latest
# or ?per_page=1 points at a tag whose assets aren't published yet.
Say "Resolving llama.cpp release..."

try {
  $recent = Invoke-RestMethod -Uri "$GitHubApi`?per_page=15" -UseBasicParsing
} catch {
  Die "Could not reach GitHub API: $_"
}

if (-not $recent -or $recent.Count -eq 0) {
  Die "No releases returned by GitHub API."
}

# Build the list of asset name patterns we'll accept for this machine.
if ($GpuTag) {
  $assetPatterns = @(
    "llama-*-bin-win-$GpuTag-$ArchTag.zip"
  )
  $runtimePattern = "cudart-llama-bin-win-$GpuTag-$ArchTag.zip"
} else {
  $assetPatterns = @(
    "llama-*-bin-win-cpu-$ArchTag.zip",
    "llama-*-bin-win-avx2-$ArchTag.zip",
    "llama-*-bin-win-$ArchTag.zip"
  )
  $runtimePattern = $null
}

$Tag = $null
$assetName = $null
$runtimeAssetName = $null

foreach ($rel in $recent) {
  if (-not $rel.tag_name -or -not $rel.assets) { continue }
  if ($rel.assets.Count -eq 0) { continue }

  # Prefer releases that actually contain an asset matching our patterns.
  foreach ($pattern in $assetPatterns) {
    $match = $rel.assets | Where-Object { $_.name -like $pattern } | Select-Object -First 1
    if ($match) {
      $Tag = $rel.tag_name
      $assetName = $match.name
      if ($runtimePattern) {
        $rt = $rel.assets | Where-Object { $_.name -like $runtimePattern } | Select-Object -First 1
        if ($rt) { $runtimeAssetName = $rt.name }
      }
      break
    }
  }

  if ($Tag) { break }
}

if (-not $Tag -or -not $assetName) {
  Warn "No prebuilt Windows asset found in the last 15 releases."
  Warn "Falling back to the newest release with any asset, and trying"
  Warn "the expected filename anyway."
  $fallback = $recent | Where-Object { $_.assets -and $_.assets.Count -gt 0 } | Select-Object -First 1
  if (-not $fallback) {
    Die "No usable llama.cpp release found on GitHub."
  }
  $Tag = $fallback.tag_name
  if ($GpuTag) {
    $assetName = "llama-$Tag-bin-win-$GpuTag-$ArchTag.zip"
  } else {
    $assetName = "llama-$Tag-bin-win-cpu-$ArchTag.zip"
  }
}

Ok "Release: $Tag"
Say "Selected asset: $assetName"
if ($runtimeAssetName) { Say "CUDA runtime: $runtimeAssetName" }

# ---------- Download and extract ----------
$tmpDir = Join-Path $env:TEMP "llama-setup-$(Get-Random)"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

$url = "$GitHubDl/$Tag/$assetName"
$zipPath = Join-Path $tmpDir $assetName

Say "Downloading $assetName ..."
try {
  Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
} catch {
  Die "Download failed for $url`n$($_.Exception.Message)"
}

Say "Extracting into $BinDir ..."
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
Expand-Archive -Path $zipPath -DestinationPath $tmpDir -Force

# Flatten any versioned subdirectory
$extracted = Get-ChildItem -Path $tmpDir -Directory | Where-Object { $_.Name -like "llama-*" } | Select-Object -First 1
if ($extracted) {
  Copy-Item -Path (Join-Path $extracted.FullName "*") -Destination $BinDir -Recurse -Force
} else {
  Copy-Item -Path (Join-Path $tmpDir "*") -Destination $BinDir -Recurse -Force
}

# Optional: separate CUDA runtime DLLs
if ($runtimeAssetName) {
  $runtimeUrl = "$GitHubDl/$Tag/$runtimeAssetName"
  $runtimeZip = Join-Path $tmpDir $runtimeAssetName
  Say "Downloading CUDA runtime DLLs ..."
  try {
    Invoke-WebRequest -Uri $runtimeUrl -OutFile $runtimeZip -UseBasicParsing -ErrorAction Stop
    Expand-Archive -Path $runtimeZip -DestinationPath $BinDir -Force
    Ok "CUDA runtime DLLs installed"
  } catch {
    Say "No separate CUDA runtime archive for this release (driver-provided DLLs will be used)."
  }
}

Remove-Item -Path $tmpDir -Recurse -Force

# ---------- Verify ----------
$serverExe = Join-Path $BinDir "llama-server.exe"
if (-not (Test-Path $serverExe)) {
  Die "llama-server.exe not found in $BinDir. Archive may have an unexpected layout."
}
$rpcExe = Join-Path $BinDir "ggml-rpc-server.exe"
if (Test-Path $rpcExe) {
  Ok "Found llama-server.exe and ggml-rpc-server.exe"
} else {
  Warn "ggml-rpc-server.exe not found; the RPC backend may not be in this build."
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
  $modelsReadme = @"
Drop your .gguf files in this folder.

Recommended starters:
  - Qwen2.5-0.5B-Instruct Q4_K_M   (~350 MB)
  - Qwen2.5-1.5B-Instruct Q4_K_M   (~1 GB)
  - Llama-3.2-3B-Instruct Q4_K_M   (~2 GB)
"@
  $modelsReadme | Out-File -Encoding utf8 (Join-Path $ModelsDir "README.txt")
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
Write-Host "  Build    : windows / $ArchTag$gpuSummary"
Write-Host "  Release  : $Tag"
Write-Host ""
Write-Host "You can now run this machine as a host, a worker, or both."
Write-Host ""
Write-Host "  Start as host (serves the API + chat UI):"
Write-Host "    .\scripts\run-host.ps1"
Write-Host ""
Write-Host "  Start as worker (joins another machine's cluster):"
Write-Host "    .\scripts\run-worker.ps1 -ApiUrl http://<host-ip>:8000"
Write-Host ""
Write-Host "  Launch the desktop GUI client:"
Write-Host "    .\scripts\run-gui.ps1"
Write-Host ""
Write-Host "To activate the virtual environment in this PowerShell window:" -ForegroundColor Cyan
Write-Host ""
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "If PowerShell refuses to run it, allow scripts for this session first:" -ForegroundColor Cyan
Write-Host ""
Write-Host "    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass" -ForegroundColor White
Write-Host ""
Write-Host "Once activated, your prompt shows '(.venv)' and 'python' resolves" -ForegroundColor Cyan
Write-Host "to the venv interpreter. Run 'deactivate' to leave it." -ForegroundColor Cyan
Write-Host ""
