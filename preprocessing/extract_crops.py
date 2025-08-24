# preprocessing/extract_crops.py

import argparse
import json
import os
from os import cpu_count
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import real_videos_paths, fake_videos_paths, test_real_videos_paths, test_fake_videos_paths, output_path

from pathlib import Path
from functools import partial
from multiprocessing.pool import Pool
import cv2
from tqdm import tqdm
from utils import get_all_video_paths

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def extract_video(video, root_output_dir):
    try:
        video_id = os.path.splitext(os.path.basename(video))[0]
        bboxes_path = os.path.join(root_output_dir, "boxes", f"{video_id}.json")

        if not os.path.exists(bboxes_path) or not os.path.exists(video):
            return

        with open(bboxes_path, "r") as bbox_f:
            bboxes_dict = json.load(bbox_f)

        capture = cv2.VideoCapture(video)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for i in range(frames_num):
            capture.grab()
            success, frame = capture.retrieve()
            if not success or str(i) not in bboxes_dict:
                continue

            bboxes = bboxes_dict[str(i)]
            if bboxes is None:
                continue

            for j, bbox in enumerate(bboxes):
                # Step 1: Scale the coordinates by 2 to match the full-resolution frame.
                xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
                
                # --- THE ROBUST FIX IS HERE ---
                # Step 2: Calculate the width and height of the detected rectangle.
                w = xmax - xmin
                h = ymax - ymin
                
                # Step 3: Calculate the padding needed to make the crop a square.
                p_h = 0
                p_w = 0
                # If the box is taller than it is wide, add horizontal padding.
                if h > w:
                    p_w = (h - w) // 2
                # If the box is wider than it is tall, add vertical padding.
                elif w > h:
                    p_h = (w - h) // 2
                
                # Step 4: Apply the padding to crop a square region.
                # The `max(..., 0)` prevents cropping outside the image boundaries.
                crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                # --- END OF FIX ---

                if crop.size == 0:
                    continue

                crops_dir = os.path.join(root_output_dir, "crops", video_id)
                os.makedirs(crops_dir, exist_ok=True)
                cv2.imwrite(os.path.join(crops_dir, f"{i}_{j}.png"), crop)

    except Exception as e:
        print(f"Error processing video {video}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', default=cpu_count() - 2, type=int, help='Number of processes for multiprocessing')
    opt = parser.parse_args()
    print(opt)

    print("Finding all videos for crop extraction (train, val, and test)...")
    all_real_dirs = real_videos_paths + test_real_videos_paths
    all_fake_dirs = fake_videos_paths + test_fake_videos_paths
    videos_to_process = get_all_video_paths(all_real_dirs, all_fake_dirs)
    print(f"Found {len(videos_to_process)} total videos to process.")

    with Pool(processes=opt.processes) as p:
        with tqdm(total=len(videos_to_process)) as pbar:
            for _ in p.imap_unordered(partial(extract_video, root_output_dir=output_path), videos_to_process):
                pbar.update()