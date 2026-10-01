param([string]$Python = "py -3.13")

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Repo '.venv'

if (-not (Test-Path -LiteralPath $Venv)) {
    $PythonParts = $Python -split " "
    $PythonExe = $PythonParts[0]
    $PythonArgs = @($PythonParts | Select-Object -Skip 1)
    & $PythonExe @PythonArgs -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Virtual-environment creation failed with exit code $LASTEXITCODE." }
}
$Python = Join-Path $Venv 'Scripts\python.exe'
& $Python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 'Python 3.13 is required')"
if ($LASTEXITCODE -ne 0) { throw "Python 3.13 is required by this bundle." }
& $Python -m pip install -r (Join-Path $Repo 'requirements\bootstrap.txt')
if ($LASTEXITCODE -ne 0) { throw "Bootstrap-tool installation failed with exit code $LASTEXITCODE." }
& $Python -m pip install -r (Join-Path $Repo 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw "Requirement installation failed with exit code $LASTEXITCODE." }
& $Python -m pip install --no-deps -e "$Repo"
if ($LASTEXITCODE -ne 0) { throw "Editable package installation failed with exit code $LASTEXITCODE." }
& $Python (Join-Path $Repo 'scripts\one_click.py')
if ($LASTEXITCODE -ne 0) { throw "Reproduction failed with exit code $LASTEXITCODE." }
