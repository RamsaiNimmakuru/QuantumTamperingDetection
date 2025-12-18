import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils_ela import batch_generate_ela

RAW_DIR = "data/raw/CASIA"
OUT_DIR = "data/processed/ELA"

if __name__ == "__main__":
    batch_generate_ela(RAW_DIR, OUT_DIR, {"Au": "authentic", "Tp": "tampered"})
