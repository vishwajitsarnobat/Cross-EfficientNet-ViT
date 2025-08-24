# test.py
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, Manager
import yaml
import argparse

from config import test_real_videos_paths, test_fake_videos_paths, architecture_config_path, workers
from utils import get_video_paths_and_labels, custom_round, custom_video_round
from cross_efficient_vit import CrossEfficientViT
from albumentations import Compose, Resize # <-- Import Resize directly

def save_roc_curve(correct_labels, preds, model_name):
    # ... (this function is unchanged)
    # ...
    pass # Keep the function as it was

def create_val_transform(size):
    """
    Creates a simple, robust transformation pipeline that force-resizes
    every image to the exact target size.
    """
    return Compose([
        Resize(height=size, width=size)
    ])

def read_test_frames(video_path_label_tuple, videos_list, image_size):
    video_path, label = video_path_label_tuple
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    crops_dir = os.path.join("processed_data", "crops", video_id)

    if not os.path.exists(crops_dir):
        return

    frames = []
    transform = create_val_transform(image_size)
    for frame_file in os.listdir(crops_dir):
        frame_path = os.path.join(crops_dir, frame_file)
        image = cv2.imread(frame_path, cv2.IMREAD_COLOR)

        if image is None:
            continue
        try:
            transformed_image = transform(image=image)['image']
            frames.append(transformed_image)
        except Exception as e:
            print(f"Warning: Failed to transform image {frame_path}. Error: {e}. Skipping.")

    if frames:
        videos_list.append({'frames': frames, 'label': label, 'name': video_id})

# --- Main body is now robust ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str, help='Path to the trained model checkpoint.')
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for prediction.")
    opt = parser.parse_args()
    print(opt)

    with open(architecture_config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    
    model = CrossEfficientViT(config=config)
    model.load_state_dict(torch.load(opt.model_path))
    model.eval()
    model = model.cuda()
    model_name = os.path.splitext(os.path.basename(opt.model_path))[0]

    print("Loading test video paths...")
    test_videos_paths = get_video_paths_and_labels(test_real_videos_paths, test_fake_videos_paths)
    
    mgr = Manager()
    videos_data = mgr.list()

    print("Reading test frames...")
    image_size = config['model']['image-size']
    with Pool(processes=workers) as p:
        with tqdm(total=len(test_videos_paths)) as pbar:
            for _ in p.imap_unordered(partial(read_test_frames, videos_list=videos_data, image_size=image_size), test_videos_paths):
                pbar.update()

    predictions = []
    correct_labels = []
    video_count = len(videos_data)

    with torch.no_grad():
        for video_info in tqdm(videos_data, desc="Predicting on test videos"):
            video_frames = video_info['frames']
            frame_preds = []
            
            for i in range(0, len(video_frames), opt.batch_size):
                batch = video_frames[i : i + opt.batch_size]
                if not batch: continue

                tensor_batch = torch.from_numpy(np.array(batch)).cuda().float()

                if tensor_batch.shape[-1] == 3:
                    tensor_batch = tensor_batch.permute(0, 3, 1, 2)
                
                pred = model(tensor_batch)
                frame_preds.extend(torch.sigmoid(pred).cpu().numpy().flatten())

            if frame_preds:
                video_pred = custom_video_round(frame_preds)
                predictions.append(video_pred)
                correct_labels.append(video_info['label'])

    print(f"\nSuccessfully processed {len(correct_labels)} out of {video_count} videos.")

    # --- Metrics ---
    if not correct_labels:
        print("Could not process any videos. No metrics to calculate.")
    else:
        accuracy = metrics.accuracy_score(correct_labels, custom_round(np.asarray(predictions)))
        f1 = metrics.f1_score(correct_labels, custom_round(np.asarray(predictions)))
        print(f"\n--- Test Results for {model_name} ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # save_roc_curve(correct_labels, predictions, model_name) # You can uncomment this if you fix the function
        cm = metrics.confusion_matrix(correct_labels, custom_round(np.asarray(predictions)))
        print("Confusion Matrix:")
        print(cm)