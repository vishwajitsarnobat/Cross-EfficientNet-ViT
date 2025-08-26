import argparse
import json
import os
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
    """
    Extracts face crops from a single video file based on pre-computed bounding boxes.
    """
    try:
        video_id = os.path.splitext(os.path.basename(video))[0]
        bboxes_path = os.path.join(root_output_dir, "boxes", f"{video_id}.json")

        # Gracefully skip if the video or its bounding box file doesn't exist.
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
                # Scale the coordinates by 2 to match the full-resolution frame.
                xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
                
                # Calculate the width and height of the detected rectangle.
                w = xmax - xmin
                h = ymax - ymin
                
                # Calculate the padding needed to make the crop a square.
                p_h = 0
                p_w = 0
                if h > w:
                    p_w = (h - w) // 2
                elif w > h:
                    p_h = (w - h) // 2
                
                # Apply the padding to crop a square region, preventing out-of-bounds access.
                crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]

                if crop.size == 0:
                    continue

                crops_dir = os.path.join(root_output_dir, "crops", video_id)
                os.makedirs(crops_dir, exist_ok=True)
                cv2.imwrite(os.path.join(crops_dir, f"{i}_{j}.png"), crop)

    except Exception as e:
        print(f"Error processing video {video}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', default=os.cpu_count() - 2, type=int, help='Number of processes for multiprocessing')
    opt = parser.parse_args()
    print(opt)

    print("Finding all videos for crop extraction (train, val, and test)...")
    all_real_dirs = real_videos_paths + test_real_videos_paths
    all_fake_dirs = fake_videos_paths + test_fake_videos_paths
    all_videos = get_all_video_paths(all_real_dirs, all_fake_dirs)
    print(f"Found {len(all_videos)} total videos.")

    # --- EFFICIENT SKIPPING LOGIC ---
    crops_base_dir = os.path.join(output_path, "crops")
    
    # Filter the list to only include videos that haven't been processed.
    unprocessed_videos = []
    for video_path in all_videos:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        # A video is considered processed if its crop directory exists.
        if not os.path.isdir(os.path.join(crops_base_dir, video_id)):
            unprocessed_videos.append(video_path)

    skipped_count = len(all_videos) - len(unprocessed_videos)
    if skipped_count > 0:
        print(f"Skipping {skipped_count} videos for which crop directories already exist.")
    
    if not unprocessed_videos:
        print("All video crops have already been extracted. Exiting.")
        exit()
        
    print(f"Found {len(unprocessed_videos)} new videos to extract crops from.")
    # --- END OF SKIPPING LOGIC ---

    # Use the filtered list of unprocessed videos for multiprocessing.
    with Pool(processes=opt.processes) as p:
        with tqdm(total=len(unprocessed_videos), desc="Extracting crops") as pbar:
            for _ in p.imap_unordered(partial(extract_video, root_output_dir=output_path), unprocessed_videos):
                pbar.update()