# Try to find available Python command
$pythonCandidates = @("python", "python3", "py")
$PYTHON = $null

foreach ($cmd in $pythonCandidates) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        $PYTHON = $cmd
        break
    }
}

if (-not $PYTHON) {
    Write-Host "Python interpreter not found!"
    exit 1
}

Write-Host "Using Python: $PYTHON"

Write-Host "Running 1_make_sentences.py..."
& $PYTHON 1_make_sentences.py

Write-Host "Running 2_translate.py..."
& $PYTHON 2_translate.py

Write-Host "Running 3_check_translations.py..."
& $PYTHON 3_check_translations.py

Write-Host "Running 4_evaluate.py..."
& $PYTHON 4_evaluate.py

Write-Host "All scripts completed."
