Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Fetching latest changes from GitHub..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
git pull

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "1. Running Data Pipeline..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
jupyter nbconvert --execute --to notebook --inplace 01_data_pipeline.ipynb

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "2. Running Model Training (QLoRA)..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
jupyter nbconvert --execute --to notebook --inplace 02_train_qlora.ipynb

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "3. Running Inference & Creating Submission..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
jupyter nbconvert --execute --to notebook --inplace 03_inference_and_submit.ipynb

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "Pipeline Complete! Your submission file is ready." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
