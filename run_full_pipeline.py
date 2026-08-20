import subprocess
import sys
import os

def run_command(command, description):
    print("=" * 60)
    print(f"{description}...")
    print("=" * 60)
    
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

def run_notebook(notebook_name, description):
    # 1. Convert notebook to python script so we can stream output in real time
    convert_cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "script", notebook_name]
    run_command(convert_cmd, f"Converting {notebook_name} to Python script")
    
    # 2. Run the generated script using IPython (so that %pip and ! commands work normally)
    script_name = notebook_name.replace('.ipynb', '.py')
    run_cmd = [sys.executable, "-m", "IPython", script_name]
    run_command(run_cmd, description)

def main():
    # Install IPython if missing (required to run script with magic commands)
    subprocess.call([sys.executable, "-m", "pip", "install", "-q", "ipython"])

    # 1. Fetch latest changes from github
    try:
        run_command(["git", "pull"], "Fetching latest changes from GitHub")
    except FileNotFoundError:
        print("[WARNING] Git is not installed or not in PATH. Skipping git pull.")

    # 2. Create local working directory (was failing because /kaggle/working didn't exist)
    os.makedirs('working', exist_ok=True)

    # 3. Run Data Pipeline
    run_notebook("01_data_pipeline.ipynb", "1. Running Data Pipeline (Streaming Output)")

    # 4. Run QLoRA Training
    run_notebook("02_train_qlora.ipynb", "2. Running Model Training (Streaming Output & Checkpoints)")

    # 5. Run Inference and Submission
    run_notebook("03_inference_and_submit.ipynb", "3. Running Inference & Creating Submission")

    print("=" * 60)
    print("Pipeline Complete! Your submission file is ready in the 'working/' folder.")
    print("=" * 60)

if __name__ == "__main__":
    main()
