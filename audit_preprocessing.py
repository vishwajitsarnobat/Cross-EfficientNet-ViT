# audit_preprocessing.py
import os
import json
import csv
from tqdm import tqdm
from termcolor import cprint

import config

def run_audit():
    """
    Scans the preprocessed 'boxes' directory to generate an audit report.
    The report details how many faces were found in each source media file.
    """
    boxes_dir = os.path.join(config.output_path, "boxes")
    # Save the report in the mode-specific results directory
    output_csv_path = os.path.join(config.results_path, "preprocessing_audit_report.csv")

    cprint("\n--- Starting Preprocessing Audit ---", "cyan", attrs=['bold'])

    if not os.path.exists(boxes_dir):
        cprint(f"Error: The 'boxes' directory was not found at '{boxes_dir}'.", "red")
        cprint("Please run the training script with 'run_preprocessing = True' at least once.", "red")
        return

    json_files = [f for f in os.listdir(boxes_dir) if f.endswith('.json')]

    if not json_files:
        cprint(f"Warning: No JSON files found in '{boxes_dir}'. Nothing to audit.", "yellow")
        return

    cprint(f"Found {len(json_files)} JSON files to audit in '{boxes_dir}'.")
    
    # --- Data Extraction ---
    audit_data = []
    for json_file in tqdm(json_files, desc="Auditing JSON files"):
        file_path = os.path.join(boxes_dir, json_file)
        media_id = os.path.splitext(json_file)[0]

        try:
            # Check for non-empty file to avoid JSONDecodeError on empty "{}" files
            if os.path.getsize(file_path) > 2:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Count total number of bounding boxes (faces) across all frames
                total_faces = sum(len(bboxes) for bboxes in data.values() if bboxes)
            else:
                total_faces = 0
            
            audit_data.append([media_id, total_faces])

        except Exception as e:
            cprint(f"\nWarning: Could not process file '{json_file}'. Error: {e}", "yellow")
            audit_data.append([media_id, "ERROR"])

    # --- Report Generation ---
    # Sort data by the number of faces found (ascending) to easily see failures
    audit_data.sort(key=lambda row: row[1] if isinstance(row[1], int) else float('inf'))

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["media_id", "faces_found"])
        writer.writerows(audit_data)

    files_with_zero_faces = sum(1 for row in audit_data if row[1] == 0)

    cprint("\n--- Audit Complete ---", "green", attrs=['bold'])
    cprint(f"✓ Audit report saved to: {output_csv_path}", "cyan")
    cprint(f"  - Files with ZERO faces detected: {files_with_zero_faces} out of {len(audit_data)}")
    cprint("  - Open the CSV report to see a file-by-file breakdown.", "cyan")
    cprint("  - This report is crucial for cleaning your dataset.", "cyan")

if __name__ == "__main__":
    run_audit()