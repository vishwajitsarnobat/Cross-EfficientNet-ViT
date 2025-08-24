# train.py
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import os
import collections
from tqdm import tqdm
from functools import partial
import cv2 # <-- Make sure cv2 is imported
import math
import yaml
import argparse
from multiprocessing import Pool

# Import settings from the central configuration file
from config import (
    real_videos_paths, fake_videos_paths, output_path, models_path,
    resume_model_path, architecture_config_path, validation_split, workers
)
from cross_efficient_vit import CrossEfficientViT
from deepfakes_dataset import DeepFakesDataset # We will rely on its changes
from torch.optim import lr_scheduler
from utils import get_video_paths_and_labels, check_correct, get_n_params, shuffle_dataset


# --- MAJOR IMPROVEMENT No. 1: Scalable Data Loading ---
# We no longer load images into memory. We just get their file paths.
def get_frame_paths(video_path_label_tuple):
    """
    For a given video, returns a list of (frame_path, label) tuples.
    This is memory-efficient as it only deals with strings.
    """
    video_path, label = video_path_label_tuple
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    crops_dir = os.path.join(output_path, "crops", video_id)

    if not os.path.exists(crops_dir):
        return []

    frame_paths = []
    for frame_file in os.listdir(crops_dir):
        frame_paths.append((os.path.join(crops_dir, frame_file), label))
    return frame_paths


# We need to modify DeepFakesDataset to handle file paths
# Let's redefine it here for clarity, or you can modify the original file.
class DeepFakesDataset(torch.utils.data.Dataset):
    def __init__(self, frame_label_list, image_size, mode='train'):
        self.frame_label_list = frame_label_list
        self.image_size = image_size
        self.mode = mode
        
        # We rely on the albumentations transforms defined in the original deepfakes_dataset.py
        from deepfakes_dataset import DeepFakesDataset as OriginalDataset
        self.transform_lib = OriginalDataset(np.array([]), np.array([]), image_size)

    def __len__(self):
        return len(self.frame_label_list)

    def __getitem__(self, index):
        frame_path, label = self.frame_label_list[index]

        # Load image from disk
        image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if image is None:
            # Handle corrupted image by returning a dummy tensor
            print(f"Warning: Could not read {frame_path}. Returning dummy data.")
            return torch.zeros((3, self.image_size, self.image_size)), torch.tensor(0.0, dtype=torch.float32)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Albumentations expects RGB

        if self.mode == 'train':
            transform = self.transform_lib.create_train_transforms(self.image_size)
        else:
            transform = self.transform_lib.create_val_transform(self.image_size)
        
        image = transform(image=image)['image']
        
        # Transpose from HWC to CHW format for PyTorch
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.float32)


# Main body
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- MAJOR IMPROVEMENT No. 2: Separate LR for Fine-tuning ---
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', default=15, type=int, help='Number of training epochs.')
    parser.add_argument('--patience', type=int, default=5, help="Epochs to wait for validation loss improvement.")
    opt = parser.parse_args()
    print(opt)

    with open(architecture_config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    # --- 1. DATA PATH GATHERING (NOW MEMORY-EFFICIENT) ---
    print("Loading video paths...")
    all_videos = get_video_paths_and_labels(real_videos_paths, fake_videos_paths)
    train_videos, val_videos = train_test_split(all_videos, test_size=validation_split, random_state=42, stratify=[label for _, label in all_videos])
    
    print("Gathering frame paths (this is fast)...")
    train_frame_paths = []
    val_frame_paths = []
    
    # Use multiprocessing to quickly gather all frame paths
    with Pool(processes=workers) as p:
        # Get train paths
        results = list(tqdm(p.imap(get_frame_paths, train_videos), total=len(train_videos), desc="Scanning train frames"))
        for res in results: train_frame_paths.extend(res)
        
        # Get validation paths
        results = list(tqdm(p.imap(get_frame_paths, val_videos), total=len(val_videos), desc="Scanning val frames"))
        for res in results: val_frame_paths.extend(res)

    print(f"Found {len(train_frame_paths)} training frames and {len(val_frame_paths)} validation frames.")
    
    # --- 2. DATASET PREPARATION ---
    train_counters = collections.Counter(label for _, label in train_frame_paths)
    class_weights = train_counters[0] / train_counters[1] if train_counters[1] > 0 else 1.0
    print(f"Training distribution: {train_counters}")
    print(f"Using class weight for FAKE class: {class_weights:.2f}")

    # Create PyTorch Datasets using the memory-efficient class
    train_dataset = DeepFakesDataset(train_frame_paths, config['model']['image-size'], mode='train')
    validation_dataset = DeepFakesDataset(val_frame_paths, config['model']['image-size'], mode='validation')

    # Create DataLoaders
    dl = DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, num_workers=workers)
    val_dl = DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=False, num_workers=workers)

    # --- 3. MODEL, OPTIMIZER, AND LOSS ---
    model = CrossEfficientViT(config=config)
    model = model.cuda()
    
    # Use the new --lr argument for the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]).cuda())
    
    starting_epoch = 0
    if resume_model_path and os.path.exists(resume_model_path):
        print(f"Loading model from: {resume_model_path} for fine-tuning.")
        model.load_state_dict(torch.load(resume_model_path))
        # We don't parse the epoch from the filename anymore, we start from 0 for fine-tuning
    else:
        print("No checkpoint loaded, training from scratch.")

    print(f"Model Parameters: {get_n_params(model)}")

    # --- 4. TRAINING LOOP ---
    not_improved_loss = 0
    previous_loss = math.inf

    for t in range(starting_epoch, opt.num_epochs):
        model.train()
        total_loss = 0
        train_correct = 0
        
        pbar = tqdm(dl, desc=f'EPOCH {t+1}/{opt.num_epochs} [TRAIN]')
        for images, labels in pbar:
            images, labels = images.cuda(), labels.cuda().unsqueeze(1)
            optimizer.zero_grad()
            
            y_pred = model(images)
            loss = loss_fn(y_pred, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            corrects, _, _ = check_correct(y_pred, labels)
            train_correct += corrects
            
            pbar.set_postfix({'loss': f'{total_loss / (pbar.n + 1):.4f}', 'acc': f'{train_correct / ((pbar.n + 1) * config["training"]["bs"]):.4f}'})

        scheduler.step()

        # Validation loop
        model.eval()
        total_val_loss = 0
        val_correct = 0
        with torch.no_grad():
            pbar_val = tqdm(val_dl, desc=f'EPOCH {t+1}/{opt.num_epochs} [VAL]')
            for val_images, val_labels in pbar_val:
                val_images, val_labels = val_images.cuda(), val_labels.cuda().unsqueeze(1)
                val_pred = model(val_images)
                val_loss = loss_fn(val_pred, val_labels)
                total_val_loss += val_loss.item()
                corrects, _, _ = check_correct(val_pred, val_labels)
                val_correct += corrects
                
                pbar_val.set_postfix({'val_loss': f'{total_val_loss / (pbar_val.n + 1):.4f}', 'val_acc': f'{val_correct / len(validation_dataset):.4f}'})
        
        avg_val_loss = total_val_loss / len(val_dl)
        if avg_val_loss < previous_loss:
            previous_loss = avg_val_loss
            not_improved_loss = 0
            print(f"Validation loss improved to {avg_val_loss:.4f}. Saving model.")
            os.makedirs(models_path, exist_ok=True)
            # Let's save with a more descriptive name
            torch.save(model.state_dict(), os.path.join(models_path, f"finetuned_checkpoint_{t}.pth"))
        else:
            not_improved_loss += 1
            print(f"Validation loss did not improve. Count: {not_improved_loss}/{opt.patience}")
            if not_improved_loss >= opt.patience:
                print("Stopping early due to validation loss not improving.")
                break