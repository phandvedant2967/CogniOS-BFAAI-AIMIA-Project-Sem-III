"""Master orchestration script to build and audit Version 2 Synthetic Dataset.

Steps executed:
  1. python scripts/generate_v2_data.py
  2. python scripts/import_data.py --raw-dir data/v2/raw --db-path data/v2/ehr.db
  3. python scripts/preprocess_ml.py --db-path data/v2/ehr.db --out-dir data/v2/processed
  4. python scripts/stage2b_ground_truth.py --db-path data/v2/ehr.db --gt-path data/v2/raw/dataset6_Ground_Truth_Cases.xlsx --out-path data/v2/processed/ground_truth_labels.csv
  5. python scripts/stage2c_build_ml_cases.py --db-path data/v2/ehr.db --labels-path data/v2/processed/ground_truth_labels.csv --out data/v2/processed/ml_cases.csv
  6. python scripts/stage2c_audit.py --db-path data/v2/ehr.db --labels-path data/v2/processed/ground_truth_labels.csv --cases-path data/v2/processed/ml_cases.csv
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

RAW_V2 = os.path.join(ROOT, "data", "v2", "raw")
PROC_V2 = os.path.join(ROOT, "data", "v2", "processed")
DB_V2 = os.path.join(ROOT, "data", "v2", "ehr.db")
GT_EXCEL = os.path.join(RAW_V2, "dataset6_Ground_Truth_Cases.xlsx")
LABELS_CSV = os.path.join(PROC_V2, "ground_truth_labels.csv")
CASES_CSV = os.path.join(PROC_V2, "ml_cases.csv")


def run_cmd(cmd, desc):
    print(f"\n=======================================================")
    print(f"=== STEP: {desc}")
    print(f"=======================================================")
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        print(f"ERROR: {desc} failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def main():
    print(f"Starting V2 Rebuild Pipeline...")

    # Step 1: Generate V2 Synthetic Data
    run_cmd([PYTHON, "scripts/generate_v2_data.py", "--out-dir", RAW_V2], "Generate V2 Raw Data")

    # Step 2: Import into SQLite DB
    run_cmd([PYTHON, "scripts/import_data.py", "--raw-dir", RAW_V2, "--db-path", DB_V2], "Import V2 to SQLite DB")

    # Step 3: Preprocess ML observation & episode files
    run_cmd([PYTHON, "scripts/preprocess_ml.py", "--db-path", DB_V2, "--out-dir", PROC_V2], "Preprocess ML Data")

    # Step 4: Build ground-truth labels CSV
    run_cmd([PYTHON, "scripts/stage2b_ground_truth.py", "--db-path", DB_V2, "--gt-path", GT_EXCEL, "--out-path", LABELS_CSV], "Build Ground Truth Labels")

    # Step 5: Build ML Cases dataset
    run_cmd([PYTHON, "scripts/stage2c_build_ml_cases.py", "--db-path", DB_V2, "--labels-path", LABELS_CSV, "--out", CASES_CSV], "Build ML Cases Dataset")

    # Step 6: Audit V2 Dataset
    run_cmd([PYTHON, "scripts/stage2c_audit.py", "--db-path", DB_V2, "--labels-path", LABELS_CSV, "--cases-path", CASES_CSV], "Audit V2 Dataset")

    print("\n=======================================================")
    print("V2 REBUILD PIPELINE COMPLETED SUCCESSFULLY!")
    print("=======================================================")


if __name__ == "__main__":
    main()
