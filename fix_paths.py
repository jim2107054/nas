import os
import glob

def fix_kaggle_paths():
    os.makedirs('working', exist_ok=True)
    notebooks = glob.glob('*.ipynb')
    
    for nb in notebooks:
        with open(nb, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if '/kaggle/working' in content:
            new_content = content.replace('/kaggle/working', 'working')
            with open(nb, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Fixed Kaggle paths in {nb}")

if __name__ == "__main__":
    fix_kaggle_paths()
