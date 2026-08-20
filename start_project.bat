@echo off
echo ==========================================
echo Fetching latest changes from GitHub...
echo ==========================================
git pull

echo.
echo ==========================================
echo Starting Jupyter Notebook...
echo ==========================================
jupyter notebook
