import argparse
import json
import os
import numpy as np
from typing import Type
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import real_videos_paths, fake_videos_paths, test_real_videos_paths, test_fake_videos_paths, output_path

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import face_detector
from face_detector import VideoDataset
from face_detector import VideoFaceDetector
from utils import get_all_video_paths


def process_videos(videos, detector_cls: Type[VideoFaceDetector], opt):
    """
    Processes a list of videos to detect faces and saves the bounding boxes.
    Assumes the input 'videos' list contains only videos that need processing.
    """
    detector = face_detector.__dict__[detector_cls](device="cuda:0")
    dataset = VideoDataset(videos)
    loader = DataLoader(dataset, shuffle=False, num_workers=opt.processes, batch_size=1, collate_fn=lambda x: x)
    missed_videos = []
    
    out_dir = os.path.join(output_path, "boxes")
    os.makedirs(out_dir, exist_ok=True)

    # The loop now only iterates over videos that need processing.
    for item in tqdm(loader, desc="Processing new videos"):
        result = {}
        video, indices, frames = item[0]
        
        id = os.path.splitext(os.path.basename(video))[0]
            
        batches = [frames[i:i + detector._batch_size] for i in range(0, len(frames), detector._batch_size)]

        for j, frames_batch in enumerate(batches):
            result.update({int(j * detector._batch_size) + i: b for i, b in zip(indices, detector._detect_faces(frames_batch))})

        if len(result) > 0:
            with open(os.path.join(out_dir, f"{id}.json"), "w") as f:
                json.dump(result, f)
        else:
            missed_videos.append(id)

    if len(missed_videos) > 0:
        print("\nThe detector did not find faces in the following videos:")
        for video_id in missed_videos:
            print(video_id)
        print("We suggest re-running the code and decreasing the detector threshold.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-type", help="Type of the detector", default="FacenetDetector",
                        choices=["FacenetDetector"])
    parser.add_argument("--processes", help="Number of processes", default=os.cpu_count() - 2, type=int)
    opt = parser.parse_args()
    print(opt)

    print("Finding all videos for preprocessing (train, val, and test)...")
    # Combine the training and test video directories
    all_real_dirs = real_videos_paths + test_real_videos_paths
    all_fake_dirs = fake_videos_paths + test_fake_videos_paths
    
    videos_paths = get_all_video_paths(all_real_dirs, all_fake_dirs)
    print(f"Found {len(videos_paths)} total videos.")

    # --- EFFICIENT SKIPPING LOGIC ---
    out_dir = os.path.join(output_path, "boxes")
    
    # Filter the list of videos BEFORE creating the dataset
    unprocessed_videos = []
    for video_path in videos_paths:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(out_dir, f"{video_id}.json")
        if not os.path.exists(output_file):
            unprocessed_videos.append(video_path)

    skipped_count = len(videos_paths) - len(unprocessed_videos)
    if skipped_count > 0:
        print(f"Skipping {skipped_count} videos that are already processed.")

    if not unprocessed_videos:
        print("All videos have already been processed. Exiting.")
        return

    print(f"Found {len(unprocessed_videos)} new videos to process.")
    # --- END OF SKIPPING LOGIC ---
    
    # Pass only the list of unprocessed videos to the function
    process_videos(unprocessed_videos, opt.detector_type, opt)


if __name__ == "__main__":
    main()