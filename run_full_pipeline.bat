@echo off
echo ==========================================
echo Fetching latest changes from GitHub...
echo ==========================================
git pull

echo.
echo ==========================================
echo 1. Running Data Pipeline...
echo ==========================================
jupyter nbconvert --execute --to notebook --inplace 01_data_pipeline.ipynb

echo.
echo ==========================================
echo 2. Running Model Training (QLoRA)...
echo ==========================================
jupyter nbconvert --execute --to notebook --inplace 02_train_qlora.ipynb

echo.
echo ==========================================
echo 3. Running Inference ^& Creating Submission...
echo ==========================================
jupyter nbconvert --execute --to notebook --inplace 03_inference_and_submit.ipynb

echo.
echo ==========================================
echo Pipeline Complete! Your submission file is ready.
echo ==========================================
