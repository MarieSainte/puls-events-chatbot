$env:PYTHONPATH = "src"
Write-Host "=============================================="
Write-Host "      LANCEMENT DES TESTS ET COVERAGE         "
Write-Host "=============================================="

$python_cmd = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $python_cmd = ".venv\Scripts\python.exe"
}

# Run pytest with coverage and output to text file
& $python_cmd -m pytest --cov=puls_events_chatbot --cov-report=term-missing *>&1 | Tee-Object -FilePath "coverage_results.txt"

Write-Host ""
Write-Host "Terminé ! Les résultats sont dans 'coverage_results.txt'."
