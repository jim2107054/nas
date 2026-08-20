Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Fetching latest changes from GitHub..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
git pull

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "Starting Jupyter Notebook..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
jupyter notebook
