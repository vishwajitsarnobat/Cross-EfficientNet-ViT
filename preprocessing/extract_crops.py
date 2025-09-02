# preprocessing/extract_crops.py
import json
import os
import sys
from functools import partial
from multiprocessing.pool import Pool

import cv2
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import output_path, preprocessing_workers, supported_formats

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def _extract_media_crops(media_path, root_output_dir):
    """
    Internal worker function for multiprocessing. Extracts crops from a single
    video or image file based on its pre-computed bounding box JSON file.
    """
    media_id = os.path.splitext(os.path.basename(media_path))[0]
    bboxes_path = os.path.join(root_output_dir, "boxes", f"{media_id}.json")

    # If the bbox file doesn't exist or is empty, there's nothing to crop.
    if not os.path.exists(bboxes_path):
        return
    
    with open(bboxes_path, "r") as bbox_f:
        try:
            bboxes_dict = json.load(bbox_f)
        except json.JSONDecodeError:
            # Handle cases where the JSON file might be empty or corrupted
            return
            
    if not bboxes_dict: # If the JSON contains just "{}"
        return

    # --- Mode-Dependent Frame/Image Reading ---
    # For a video, we iterate through its frames
    if media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        try:
            capture = cv2.VideoCapture(media_path)
            frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            for i in range(frames_num):
                capture.grab()
                # Crop only if a bounding box exists for this frame index
                if str(i) not in bboxes_dict:
                    continue
                
                success, frame = capture.retrieve()
                if not success:
                    continue
                
                # Process all detected faces in this frame
                _process_frame_crops(frame, i, bboxes_dict[str(i)], media_id, root_output_dir)
        finally:
            if 'capture' in locals() and capture.isOpened():
                capture.release()

    # For an image, we read it once
    elif media_path.lower().endswith(supported_formats):
        frame = cv2.imread(media_path)
        if frame is None:
            return
        
        # In the image case, bboxes are stored under the key "0"
        if "0" in bboxes_dict:
            # The frame index is 0, as it's a single image
            _process_frame_crops(frame, 0, bboxes_dict["0"], media_id, root_output_dir)

def _process_frame_crops(frame, frame_idx, bboxes, media_id, root_output_dir):
    """Helper function to extract, save, and process crops from a single frame."""
    for j, bbox in enumerate(bboxes):
        if not bbox: continue
        
        # The detector resizes images by 1/2, so we multiply by 2 to get original coordinates
        xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
        w, h = xmax - xmin, ymax - ymin
        
        # Make the crop square by adding padding
        p_h = (w - h) // 2 if w > h else 0
        p_w = (h - w) // 2 if h > w else 0
        
        # Ensure coordinates are within image bounds
        crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]

        if crop.size == 0:
            continue

        # Save the crop
        crops_dir = os.path.join(root_output_dir, "crops", media_id)
        os.makedirs(crops_dir, exist_ok=True)
        # Naming convention: {frame_index}_{crop_index}.png
        cv2.imwrite(os.path.join(crops_dir, f"{frame_idx}_{j}.png"), crop)

def run_crop_extraction(media_to_scan: list):
    """
    Extracts face crops for a given list of media files (videos or images).
    Correctly skips files where crops have already been generated or no faces were found.
    """
    print("--- Running Face Crop Extraction ---")
    
    media_to_process = []
    for media_path in media_to_scan:
        media_id = os.path.splitext(os.path.basename(media_path))[0]
        bbox_file = os.path.join(output_path, "boxes", f"{media_id}.json")
        crop_folder = os.path.join(output_path, "crops", media_id)
        
        # --- MODIFIED LOGIC ---
        # We only process a file if:
        # 1. Its bbox JSON exists.
        # 2. The JSON file is not empty (size > 2 bytes for "{}").
        # 3. Its crop folder has NOT yet been created.
        if os.path.exists(bbox_file) and os.path.getsize(bbox_file) > 2 and not os.path.exists(crop_folder):
            media_to_process.append(media_path)

    print(f"Scope: {len(media_to_scan)} media files. Found {len(media_to_process)} that need cropping.")

    if not media_to_process:
        print("All valid media files in the current scope have already been cropped.")
        print("--- Face Crop Extraction Complete ---")
        return
    
    # Use a multiprocessing pool to parallelize the extraction
    with Pool(processes=preprocessing_workers) as p:
        with tqdm(total=len(media_to_process), desc="Extracting Crops") as pbar:
            func = partial(_extract_media_crops, root_output_dir=output_path)
            for _ in p.imap_unordered(func, media_to_process):
                pbar.update()
    
    print("--- Face Crop Extraction Complete ---")