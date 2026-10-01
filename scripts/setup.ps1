[CmdletBinding()]
param(
    [string]$Python = "py -3.13",
    [ValidateSet("analysis", "kev")]
    [string]$Profile = "analysis",
    [string]$Models = "",
    [switch]$AllowUnpinnedModels,
    [switch]$DownloadUpstreamData,
    [switch]$RebuildInputs,
    [switch]$AllowUnpinnedData
)

$ErrorActionPreference = "Stop"
$ScriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $ScriptsDir
$Venv = Join-Path $Repo ".venv"
$PythonParts = $Python -split " "
$PythonExe = $PythonParts[0]
$PythonArgs = @($PythonParts | Select-Object -Skip 1)

if (-not (Test-Path -LiteralPath $Venv)) {
    & $PythonExe @PythonArgs -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Virtual-environment creation failed with exit code $LASTEXITCODE." }
}
$VenvPython = Join-Path $Venv "Scripts\python.exe"
& $VenvPython -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 'Python 3.13 is required')"
if ($LASTEXITCODE -ne 0) { throw "Python 3.13 is required by this bundle." }
$Requirements = if ($Profile -eq "kev") { "requirements\kev.txt" } else { "requirements\base.txt" }
if ($Models -and $Profile -ne "kev") {
    throw "Model download/rerun dependencies require -Profile kev."
}
if ($Profile -eq "kev") { $env:PIP_NO_BUILD_ISOLATION = "1" }
& $VenvPython -m pip install -r (Join-Path $Repo $Requirements)
if ($LASTEXITCODE -ne 0) { throw "Requirement installation failed with exit code $LASTEXITCODE." }
if ($DownloadUpstreamData -or $RebuildInputs) {
    & $VenvPython -m pip install -r (Join-Path $Repo "requirements\data.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dataset requirement installation failed with exit code $LASTEXITCODE." }
}
& $VenvPython -m pip install --no-deps -e $Repo
if ($LASTEXITCODE -ne 0) { throw "Editable package installation failed with exit code $LASTEXITCODE." }

if ($RebuildInputs) {
    $RebuildArgs = @((Join-Path $Repo "scripts\rebuild_inputs.py"), "--download")
    if ($AllowUnpinnedData) { $RebuildArgs += "--allow-unpinned" }
    & $VenvPython @RebuildArgs
    if ($LASTEXITCODE -ne 0) { throw "Upstream reconstruction failed with exit code $LASTEXITCODE." }
} elseif ($DownloadUpstreamData) {
    $SourceSet = if ($AllowUnpinnedData) { "all" } else { "pinned" }
    $DataArgs = @((Join-Path $Repo "scripts\download_datasets.py"), "--sources", $SourceSet)
    if ($AllowUnpinnedData) { $DataArgs += "--allow-unpinned" }
    & $VenvPython @DataArgs
    if ($LASTEXITCODE -ne 0) { throw "Dataset download failed with exit code $LASTEXITCODE." }
}
if ($Models) {
    $ModelArgs = @((Join-Path $Repo "scripts\download_models.py"), "--models", $Models)
    if ($AllowUnpinnedModels) { $ModelArgs += "--allow-unpinned" }
    & $VenvPython @ModelArgs
    if ($LASTEXITCODE -ne 0) { throw "Model download failed with exit code $LASTEXITCODE." }
}
& $VenvPython (Join-Path $Repo "scripts\verify_artifacts.py")
if ($LASTEXITCODE -ne 0) { throw "Bundle verification failed with exit code $LASTEXITCODE." }

Write-Host "Bundle ready. Python: $VenvPython"
