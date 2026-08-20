@echo off
echo ==========================================
echo Fetching latest changes from GitHub...
echo ==========================================
git pull

echo.
echo ==========================================
echo 1. Running Data Pipeline...
echo ==========================================
python -m jupyter nbconvert --execute --to notebook --inplace 01_data_pipeline.ipynb

echo.
echo ==========================================
echo 2. Running Model Training (QLoRA)...
echo ==========================================
python -m jupyter nbconvert --execute --to notebook --inplace 02_train_qlora.ipynb

echo.
echo ==========================================
echo 3. Running Inference ^& Creating Submission...
echo ==========================================
python -m jupyter nbconvert --execute --to notebook --inplace 03_inference_and_submit.ipynb

echo.
echo ==========================================
echo Pipeline Complete! Your submission file is ready.
echo ==========================================
