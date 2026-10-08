# Run from any working directory. Training and report generation stop on failure.
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $taskPython = Join-Path $PSScriptRoot '..\.runtime\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) {
        $taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    }
    if (-not (Test-Path -LiteralPath $taskPython)) {
        throw 'Create a virtual environment and install requirements.txt first; see README.md.'
    }
    & $taskPython test_core.py
    if ($LASTEXITCODE -ne 0) { throw 'Core checks failed.' }
    & $taskPython download_glove.py
    if ($LASTEXITCODE -ne 0) { throw 'GloVe download failed.' }
    $taskModel = 'google-bert/bert-base-uncased'
    if (Test-Path -LiteralPath 'assets\bert-base-uncased\model.safetensors') {
        $taskModel = 'assets/bert-base-uncased'
    }
    & $taskPython experiment.py --glove assets/glove.6B.100d.txt --bert-model $taskModel
    if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; see output above.' }
    & $taskPython analyze_results.py
    if ($LASTEXITCODE -ne 0) { throw 'Additional analysis failed.' }
    & $taskPython validate_results.py
    if ($LASTEXITCODE -ne 0) { throw 'Result verification failed.' }
    & $taskPython report.py
    if ($LASTEXITCODE -ne 0) { throw 'Report creation failed.' }
} finally {
    Pop-Location
}
