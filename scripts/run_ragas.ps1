$env:PYTHONPATH = "src"

Write-Host "=============================================="
Write-Host "       LANCEMENT DE L'ÉVALUATION RAGAS        "
Write-Host "=============================================="
poetry run python evaluation/evaluate_rag.py *>&1 | Tee-Object -FilePath "ragas_results.txt"
Write-Host ""
Write-Host "Évaluation terminée ! Vous pouvez consulter 'ragas_results.txt'."
