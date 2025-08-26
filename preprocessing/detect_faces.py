# preprocessing/detect_faces.py
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from . import face_detector
from .face_detector import VideoDataset
from config import output_path, preprocessing_workers
from torch.utils.data import DataLoader
from tqdm import tqdm

def run_face_detection(videos_to_scan: list, detector_cls: str = "FacenetDetector"):
    """
    Detects faces in a given list of videos and saves bounding boxes.
    This version correctly skips videos in the list that are already processed.
    """
    print("--- Running Face Detection ---")
    
    out_dir = os.path.join(output_path, "boxes")
    os.makedirs(out_dir, exist_ok=True)

    # Pre-filter to find which of the *provided* videos need processing.
    videos_to_process = []
    for video_path in videos_to_scan:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        if not os.path.exists(os.path.join(out_dir, f"{video_id}.json")):
            videos_to_process.append(video_path)
    
    print(f"Scope: {len(videos_to_scan)} videos. Found {len(videos_to_process)} that need processing.")
    
    if not videos_to_process:
        print("All videos in the current scope have already been processed for face detection.")
        print("--- Face Detection Complete ---")
        return

    detector = face_detector.__dict__[detector_cls](device="cuda:0")
    dataset = VideoDataset(videos_to_process)
    loader = DataLoader(dataset, shuffle=False, num_workers=preprocessing_workers, batch_size=1, collate_fn=lambda x: x)
    
    missed_videos = []

    for item in tqdm(loader, desc="Detecting Faces"):
        result = {}
        video, indices, frames = item[0]
        video_id = os.path.splitext(os.path.basename(video))[0]
            
        batches = [frames[i:i + detector._batch_size] for i in range(0, len(frames), detector._batch_size)]

        for j, frames_batch in enumerate(batches):
            detections = detector._detect_faces(frames_batch)
            result.update({int(j * detector._batch_size) + i: b for i, b in zip(indices, detections)})

        if len(result) > 0:
            with open(os.path.join(out_dir, f"{video_id}.json"), "w") as f:
                json.dump(result, f)
        else:
            missed_videos.append(video_id)

    if len(missed_videos) > 0:
        print(f"\nWarning: The detector did not find faces in {len(missed_videos)} videos.")
    
    print("--- Face Detection Complete ---")