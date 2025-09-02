# preprocessing/detect_faces.py
import json
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from . import face_detector
from .face_detector import MediaDataset
from torch.utils.data import DataLoader

def run_face_detection(media_to_scan: list, detector_cls: str = "FacenetDetector"):
    print("--- Running Face Detection ---")
    out_dir = os.path.join(config.output_path, "boxes")
    os.makedirs(out_dir, exist_ok=True)

    media_to_process = []
    print("Checking for media files that need processing...")
    for media_path in tqdm(media_to_scan, desc="Pre-filtering media"):
        media_id = os.path.splitext(os.path.basename(media_path))[0]
        if not os.path.exists(os.path.join(out_dir, f"{media_id}.json")):
            media_to_process.append(media_path)
    
    print(f"Scope: {len(media_to_scan)} media files. Found {len(media_to_process)} that need processing.")
    
    if not media_to_process:
        print("All media files in the current scope have already been processed for face detection.")
        print("--- Face Detection Complete ---")
        return

    detector = face_detector.__dict__[detector_cls](device=config.device)
    
    dataset = MediaDataset(media_to_process)
    loader = DataLoader(dataset, shuffle=False, num_workers=config.preprocessing_workers, batch_size=1, collate_fn=lambda x: x)
    
    missed_files = []

    for item in tqdm(loader, desc="Detecting Faces"):
        if not item: continue
        result = {}
        media_path, indices, frames = item[0]
        media_id = os.path.splitext(os.path.basename(media_path))[0]
        output_json_path = os.path.join(out_dir, f"{media_id}.json")

        if frames:
            batches = [frames[i:i + detector._batch_size] for i in range(0, len(frames), detector._batch_size)]
            for j, frames_batch in enumerate(batches):
                detections = detector._detect_faces(frames_batch)
                valid_detections = {
                    int(indices[j * detector._batch_size + i]): b
                    for i, b in enumerate(detections) if b is not None and len(b) > 0
                }
                result.update(valid_detections)

        if not result:
            missed_files.append(media_id)
        
        with open(output_json_path, "w") as f:
            json.dump(result, f)

    if len(missed_files) > 0:
        print(f"\nWarning: The detector did not find faces in {len(missed_files)} files (or they were empty).")
        print("An empty JSON file was created for each to prevent reprocessing.")
    
    print("--- Face Detection Complete ---")