<#
.SYNOPSIS
    Rehearse the Kaggle run on this machine, on Kaggle's old pinned library versions.

.DESCRIPTION
    Kaggle can hand the notebook its pinned 2023 image: Python 3.10 and
    PyTorch 2.0. Two earlier runs died there on post-2.0 APIs, after the
    pre-flight cells had already used GPU time. This script rebuilds that
    environment on the CPU and runs the code in it:

      1. a fresh Python 3.10 venv (uv);
      2. torch 2.0.0 + torchvision 0.15.1 (CPU builds) and the image's other
         packages, at the versions in $ImagePins below;
      3. the notebook's own extras install, read out of kaggle_runner.ipynb and
         run with pip exactly as the notebook runs it. If that install drags
         numpy or torch to another version, the step fails and says which;
      4. every .py in src/ and scripts/ compiled by Python 3.10;
      5. the GradScaler fallback resolved on torch 2.0;
      6. scripts/smoke_test.py --tiny: all six models through the real
         training loop, interrupted and resumed, plus the E2/E3/E4 paths;
      7. with -WithQueue, scripts/queue_selftest.py too: the real queue on toy
         runs, E5/E6/E7 included (adds ~5-10 min).

    What a CPU cannot cover: CUDA. fp16 autocast and the GradScaler's actual
    scaling only run on a GPU -- that is the notebook's SMOKE = True run.

    The pins reconstruct Kaggle's 2023 image; they are not read from it.
    torch, torchvision, numpy and timm are the versions that image reported;
    the rest are contemporaneous releases. The notebook's install cell prints
    the versions Kaggle actually ends up with -- correct $ImagePins from that
    log if they differ.

    Writes results/rehearsal.json (the record) and results/_rehearsal/
    (scratch, gitignored). Exits 0 only if every step passed.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\rehearse.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\rehearse.ps1 -WithQueue
#>
[CmdletBinding()]
param(
    # Any interpreter spec uv understands; Kaggle's pinned image runs 3.10.
    [string]$Python = "3.10",
    # Recreated from scratch on every run, so each rehearsal starts clean.
    [string]$VenvDir = (Join-Path $env:LOCALAPPDATA "showdown-rehearse\py310-torch2.0"),
    [switch]$WithQueue
)

# Native tools (uv, pip, python) report through exit codes, checked
# explicitly below. "Stop" would turn their ordinary stderr output (uv's
# progress lines) into terminating errors under Windows PowerShell 5.1.
$ErrorActionPreference = "Continue"

$Repo = Split-Path -Parent $PSScriptRoot
$Scratch = "results/_rehearsal"
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# --- Kaggle's pinned 2023 image ---------------------------------------------
$TorchIndex = "https://download.pytorch.org/whl/cpu"
$TorchPins = @("torch==2.0.0", "torchvision==0.15.1")       # CUDA builds on Kaggle
$ImagePins = @(
    "numpy==1.24.3", "timm==0.9.12",
    # The image's setuptools. uv seeds today's, which no longer ships
    # pkg_resources -- and umap-learn 0.5.4 imports it, failing E7's UMAP step.
    "setuptools==68.2.2",
    # torch/torchvision's own dependencies (installed --no-deps above)
    "filelock==3.12.2", "typing-extensions==4.8.0", "sympy==1.12", "networkx==3.2.1",
    "jinja2==3.1.2", "requests==2.31.0", "pillow==9.5.0",
    # timm's
    "huggingface-hub==0.19.4", "safetensors==0.4.1", "pyyaml==6.0.1",
    # the rest of the scientific stack the code and the notebook extras touch
    "scipy==1.11.4", "scikit-learn==1.2.2", "pandas==2.0.3", "matplotlib==3.7.3",
    "seaborn==0.12.2", "opencv-python==4.8.1.78", "umap-learn==0.5.4", "numba==0.58.1",
    "tqdm==4.66.1", "tensorboard==2.15.1", "psutil==5.9.5"
)
# Distributions whose versions are recorded, and checked for drift.
# grad-cam requires opencv-python-headless unpinned; its 4.12+ releases need
# numpy 2, so this is the one to watch when the image's numpy is 1.x.
$Watch = @("torch", "torchvision", "numpy", "timm", "scipy", "scikit-learn", "pandas",
           "matplotlib", "seaborn", "opencv-python", "opencv-python-headless", "umap-learn",
           "numba", "pillow", "pyyaml", "tqdm", "tensorboard", "psutil", "grad-cam", "ttach",
           "setuptools")

# --- helpers -----------------------------------------------------------------
$Steps = New-Object System.Collections.Generic.List[object]
$script:Failed = $false

function Assert-Exit([string]$What) {
    if ($LASTEXITCODE -ne 0) { throw "$What exited with code $LASTEXITCODE" }
}

function Invoke-Step([string]$Name, [scriptblock]$Body) {
    if ($script:Failed) {
        $Steps.Add([ordered]@{ step = $Name; ok = $null; seconds = 0; note = "not run (an earlier step failed)" })
        return
    }
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    $t0 = Get-Date
    $note = ""
    try {
        & $Body
        $ok = $true
    } catch {
        $ok = $false
        $note = $_.Exception.Message
    }
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $Steps.Add([ordered]@{ step = $Name; ok = $ok; seconds = $secs; note = $note })
    if ($ok) {
        Write-Host "[PASS] $Name  ($secs s)" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] $Name  ($secs s): $note" -ForegroundColor Red
        $script:Failed = $true
    }
}

# Python snippets go in on stdin: Windows PowerShell 5.1 mangles double quotes
# inside native-command arguments, so `python -c "..."` is not safe here.
$VersionsPy = @'
import json, sys
from importlib import metadata
out = {"python": sys.version.split()[0]}
for name in sys.argv[1:]:
    try:
        out[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        out[name] = None
print(json.dumps(out))
'@

$ExtrasPy = @'
import json, re, sys
nb = json.load(open("kaggle_runner.ipynb", encoding="utf-8"))
for cell in nb["cells"]:
    m = re.search(r'"pip", "install", "-q",(.*?)\]', "".join(cell["source"]), re.S)
    if m:
        print(" ".join(re.findall(r'"([^"]+)"', m.group(1))))
        sys.exit(0)
sys.exit("no pip install line found in kaggle_runner.ipynb")
'@

$ScalerPy = @'
import torch
from src.train import make_grad_scaler
s = make_grad_scaler(enabled=False)
missing = [a for a in ("scale", "step", "update", "unscale_", "state_dict", "load_state_dict")
           if not hasattr(s, a)]
print(f"torch {torch.__version__}: torch.amp.GradScaler "
      f"{'present' if hasattr(torch.amp, 'GradScaler') else 'absent'} -> make_grad_scaler() "
      f"returns {type(s).__module__}.{type(s).__name__}")
assert not missing, f"scaler lacks {missing}"
'@

function Get-Versions {
    $json = $VersionsPy | & $VenvPy - @Watch
    Assert-Exit "version probe"
    return ($json | ConvertFrom-Json)
}

# --- the rehearsal -------------------------------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is required: https://docs.astral.sh/uv/" -ForegroundColor Red
    exit 1
}
Push-Location $Repo
$started = Get-Date
try {
    New-Item -ItemType Directory -Force $Scratch | Out-Null

    Invoke-Step "fresh Python $Python venv" {
        & uv venv --clear --seed -p $Python $VenvDir
        Assert-Exit "uv venv"
    }
    Invoke-Step "torch 2.0.0 + torchvision 0.15.1 (CPU builds)" {
        & uv pip install -p $VenvPy --no-deps --default-index $TorchIndex @TorchPins
        Assert-Exit "uv pip install (torch)"
    }
    Invoke-Step "Kaggle image packages at their pinned versions" {
        & uv pip install -p $VenvPy @ImagePins
        Assert-Exit "uv pip install (image pins)"
        $script:Before = Get-Versions
    }
    Invoke-Step "the notebook's extras install, run with pip as the notebook runs it" {
        $extras = ($ExtrasPy | & $VenvPy -)
        Assert-Exit "reading the extras from kaggle_runner.ipynb"
        $pkgs = $extras.Trim() -split "\s+"
        Write-Host "pip install -q $($pkgs -join ' ')"
        & $VenvPy -m pip install -q --disable-pip-version-check @pkgs
        Assert-Exit "pip install (notebook extras)"
        $script:After = Get-Versions
        $changed = @(); $added = @()
        foreach ($name in $Watch) {
            $b = $script:Before.$name; $a = $script:After.$name
            if ($b -and $a -and $b -ne $a) { $changed += "$name $b -> $a" }
            if (-not $b -and $a) { $added += "$name $a" }
        }
        if ($added) { Write-Host ("pip added: " + ($added -join ", ")) }
        if ($changed) { Write-Host ("pip changed: " + ($changed -join ", ")) -ForegroundColor Yellow }
        $drift = $changed | Where-Object { $_ -match "^(torch|torchvision|numpy) " }
        if ($drift) { throw "the notebook's install moved a pinned core package: $($drift -join ', ')" }
        $script:After.PSObject.Properties | ForEach-Object { "  {0,-14} {1}" -f $_.Name, $_.Value } |
            Write-Host
    }
    Invoke-Step "every .py in src/ and scripts/ compiles on Python $Python" {
        & $VenvPy -m compileall -q src scripts
        Assert-Exit "compileall"
    }
    Invoke-Step "GradScaler fallback on torch 2.0" {
        $ScalerPy | & $VenvPy -
        Assert-Exit "GradScaler check"
    }
    Invoke-Step "smoke_test.py --tiny (all six models + E2/E3/E4 paths, CPU)" {
        $smokeArgs = @("--tiny", "--num-workers", "2", "--out", "$Scratch/smoke",
                       "--summary", "$Scratch/smoke_test.json")
        & $VenvPy scripts/smoke_test.py @smokeArgs
        Assert-Exit "smoke_test.py --tiny"
    }
    if ($WithQueue) {
        Invoke-Step "queue_selftest.py (real queue on toy runs, E5/E6/E7 included)" {
            & $VenvPy scripts/queue_selftest.py --summary "$Scratch/queue_selftest.json"
            Assert-Exit "queue_selftest.py"
        }
    }
} finally {
    Pop-Location
}

# --- the record ---------------------------------------------------------------
$passed = -not $script:Failed
$record = [ordered]@{
    passed       = $passed
    started      = $started.ToString("yyyy-MM-ddTHH:mm:ss")
    minutes      = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
    python       = $Python
    torch_index  = $TorchIndex
    pins         = @($TorchPins + $ImagePins)
    installed    = $script:After
    # .ToArray(), not @($Steps): PowerShell 5.1 throws "Argument types do not
    # match" on @() of this list.
    steps        = $Steps.ToArray()
}
$json = $record | ConvertTo-Json -Depth 6
# UTF-8 without a BOM: Python's json module rejects the BOM PowerShell 5.1 writes.
[System.IO.File]::WriteAllText((Join-Path $Repo "results\rehearsal.json"), $json + "`n",
                               (New-Object System.Text.UTF8Encoding $false))

Write-Host ""
Write-Host "================ rehearsal summary ================"
foreach ($s in $Steps) {
    $mark = if ($s.ok -eq $true) { "PASS" } elseif ($s.ok -eq $false) { "FAIL" } else { "----" }
    Write-Host ("[{0}] {1,7}s  {2}" -f $mark, $s.seconds, $s.step)
}
if ($passed) {
    Write-Host "REHEARSAL PASSED -- written to results/rehearsal.json" -ForegroundColor Green
    exit 0
}
Write-Host "REHEARSAL FAILED -- see above; written to results/rehearsal.json" -ForegroundColor Red
exit 1
