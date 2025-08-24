# preprocessing/detect_faces.py

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
    detector = face_detector.__dict__[detector_cls](device="cuda:0")
    dataset = VideoDataset(videos)
    loader = DataLoader(dataset, shuffle=False, num_workers=opt.processes, batch_size=1, collate_fn=lambda x: x)
    missed_videos = []
    
    out_dir = os.path.join(output_path, "boxes")
    os.makedirs(out_dir, exist_ok=True)

    for item in tqdm(loader):
        result = {}
        video, indices, frames = item[0]
        
        id = os.path.splitext(os.path.basename(video))[0]

        if os.path.exists(os.path.join(out_dir, "{}.json".format(id))):
            continue
            
        batches = [frames[i:i + detector._batch_size] for i in range(0, len(frames), detector._batch_size)]

        for j, frames_batch in enumerate(batches):
            result.update({int(j * detector._batch_size) + i: b for i, b in zip(indices, detector._detect_faces(frames_batch))})

        if len(result) > 0:
            with open(os.path.join(out_dir, "{}.json".format(id)), "w") as f:
                json.dump(result, f)
        else:
            missed_videos.append(id)

    if len(missed_videos) > 0:
        print("The detector did not find faces inside the following videos:")
        for video_id in missed_videos:
            print(video_id)
        print("We suggest to re-run the code decreasing the detector threshold.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-type", help="Type of the detector", default="FacenetDetector",
                        choices=["FacenetDetector"])
    parser.add_argument("--processes", help="Number of processes", default=1, type=int)
    opt = parser.parse_args()
    print(opt)

    print("Finding all videos for preprocessing (train, val, and test)...")
    # Combine the training and test video directories
    all_real_dirs = real_videos_paths + test_real_videos_paths
    all_fake_dirs = fake_videos_paths + test_fake_videos_paths
    
    videos_paths = get_all_video_paths(all_real_dirs, all_fake_dirs)
    print(f"Found {len(videos_paths)} total videos to process.")

    process_videos(videos_paths, opt.detector_type, opt)


if __name__ == "__main__":
    main()