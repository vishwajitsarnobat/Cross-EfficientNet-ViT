# preprocessing/extract_crops.py
import json
import os
import sys
from functools import partial
from multiprocessing.pool import Pool

import cv2
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import output_path, preprocessing_workers

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def _extract_video_crops(video_path, root_output_dir):
    """Internal worker function for multiprocessing."""
    try:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        bboxes_path = os.path.join(root_output_dir, "boxes", f"{video_id}.json")

        with open(bboxes_path, "r") as bbox_f:
            bboxes_dict = json.load(bbox_f)

        capture = cv2.VideoCapture(video_path)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for i in range(frames_num):
            capture.grab()
            if str(i) not in bboxes_dict:
                continue
            
            success, frame = capture.retrieve()
            if not success or bboxes_dict.get(str(i)) is None:
                continue

            for j, bbox in enumerate(bboxes_dict[str(i)]):
                xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
                w, h = xmax - xmin, ymax - ymin
                
                p_h = (w - h) // 2 if w > h else 0
                p_w = (h - w) // 2 if h > w else 0
                
                crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]

                if crop.size == 0:
                    continue

                crops_dir = os.path.join(root_output_dir, "crops", video_id)
                os.makedirs(crops_dir, exist_ok=True)
                cv2.imwrite(os.path.join(crops_dir, f"{i}_{j}.png"), crop)
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")

def run_crop_extraction(videos_to_scan: list):
    """
    Extracts face crops for a given list of videos.
    Correctly skips videos that are already cropped.
    """
    print("--- Running Face Crop Extraction ---")
    
    videos_to_process = []
    for video_path in videos_to_scan:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        bbox_file = os.path.join(output_path, "boxes", f"{video_id}.json")
        crop_folder = os.path.join(output_path, "crops", video_id)
        if os.path.exists(bbox_file) and not os.path.exists(crop_folder):
            videos_to_process.append(video_path)

    print(f"Scope: {len(videos_to_scan)} videos. Found {len(videos_to_process)} that need cropping.")

    if not videos_to_process:
        print("All videos in the current scope have already been cropped.")
        print("--- Face Crop Extraction Complete ---")
        return
    
    with Pool(processes=preprocessing_workers) as p:
        with tqdm(total=len(videos_to_process), desc="Extracting Crops") as pbar:
            func = partial(_extract_video_crops, root_output_dir=output_path)
            for _ in p.imap_unordered(func, videos_to_process):
                pbar.update()
    
    print("--- Face Crop Extraction Complete ---")