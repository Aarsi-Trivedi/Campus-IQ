#!/usr/bin/env python3
"""
Campus IQ - Command Line Excel (.xlsx / .xls) & CSV Importer
Allows importing custom campus observations from Excel files directly into the Campus IQ pipeline.
Usage:
    python3 import_xlsx.py /path/to/your/campus_data.xlsx [--retrain] [--mode append|replace]
"""

import os
import sys
import argparse
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
MASTER_CSV = os.path.join(DATA_DIR, "campus_iq_simulated.csv")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from excel_importer import import_and_process_file
from run_pipeline import run_full_pipeline

def main():
    parser = argparse.ArgumentParser(description="Import Excel (.xlsx) campus data into Campus IQ.")
    parser.add_argument("file_path", help="Path to your Excel (.xlsx / .xls) or CSV file")
    parser.add_argument("--mode", choices=["append", "replace"], default="replace", help="Whether to append to or replace current dataset (default: replace)")
    parser.add_argument("--retrain", action="store_true", default=True, help="Automatically retrain all ML models on the new dataset")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File not found at '{args.file_path}'")
        sys.exit(1)

    print(f"\n[Importer] Reading and parsing Excel file: {args.file_path}")
    try:
        new_df = import_and_process_file(args.file_path, default_label="USER_SUBMITTED_CAMPUS_DATA")
        print(f"[Importer] Successfully parsed {len(new_df)} observations with {len(new_df.columns)} mapped features!")
    except Exception as e:
        print(f"[Importer] Error processing Excel file: {e}")
        sys.exit(1)

    if args.mode == "append" and os.path.exists(MASTER_CSV):
        existing_df = pd.read_csv(MASTER_CSV)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Re-index record IDs
        combined_df["record_id"] = [f"CIQ-{i+1:04d}" for i in range(len(combined_df))]
        combined_df.to_csv(MASTER_CSV, index=False)
        print(f"[Importer] Appended {len(new_df)} rows. Total dataset now has {len(combined_df)} records.")
    else:
        new_df["record_id"] = [f"CIQ-{i+1:04d}" for i in range(len(new_df))]
        new_df.to_csv(MASTER_CSV, index=False)
        print(f"[Importer] Master dataset updated with {len(new_df)} custom observations.")

    if args.retrain:
        print("\n[Importer] Retraining ML models, recalculating statistics & clustering on new data...")
        run_full_pipeline()
        print("\n[Importer] SUCCESS! Models and dashboard data refreshed with your custom Excel data.")
        print("View the updated dashboard at: http://localhost:8090")

if __name__ == "__main__":
    main()
