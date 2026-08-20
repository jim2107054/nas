import subprocess
import sys
import os

def run_command(command, description):
    print("=" * 50)
    print(f"{description}...")
    print("=" * 50)
    
    try:
        # Run command and pipe output to terminal in real-time
        process = subprocess.Popen(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            shell=False
        )
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[ERROR] Command failed with exit code {process.returncode}")
            sys.exit(process.returncode)
            
    except Exception as e:
        print(f"\n[ERROR] Failed to run command: {e}")
        sys.exit(1)
    
    print("\n")

def main():
    # 1. Fetch latest changes from github (requires git installed)
    try:
        run_command(["git", "pull"], "Fetching latest changes from GitHub")
    except FileNotFoundError:
        print("[WARNING] Git is not installed or not in PATH. Skipping git pull.")

    # Base jupyter nbconvert command
    nbconvert_cmd = [
        sys.executable, "-m", "jupyter", "nbconvert", 
        "--execute", "--to", "notebook", "--inplace"
    ]

    # 2. Run Data Pipeline
    run_command(nbconvert_cmd + ["01_data_pipeline.ipynb"], "1. Running Data Pipeline")

    # 3. Run QLoRA Training
    run_command(nbconvert_cmd + ["02_train_qlora.ipynb"], "2. Running Model Training (QLoRA)")

    # 4. Run Inference and Submission
    run_command(nbconvert_cmd + ["03_inference_and_submit.ipynb"], "3. Running Inference & Creating Submission")

    print("=" * 50)
    print("Pipeline Complete! Your submission file is ready.")
    print("=" * 50)

if __name__ == "__main__":
    main()
