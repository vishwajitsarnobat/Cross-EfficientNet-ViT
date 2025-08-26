import os
import re
import random
import shutil
from collections import defaultdict
from tqdm import tqdm

# ==============================================================================
# ===               PLEASE EDIT THE VARIABLES IN THIS SECTION                ===
# ==============================================================================

# List all the directories that contain your REAL videos.
REAL_VIDEO_DIRS = [
    "/mnt/data/CelebDF-V2/Celeb-real",
    "/mnt/data/CelebDF-V2/YouTube-real"
]

# The single directory that contains all your FAKE videos.
FAKE_VIDEO_DIR = "/mnt/data/CelebDF-V2/Celeb-synthesis"

# The directory where the new balanced dataset will be created.
# Two subfolders, 'train_real' and 'train_fake', will be made here.
OUTPUT_DIR = "/mnt/data/Cross-EfficientNet-data"

# ==============================================================================
# ===                  NO NEED TO EDIT BELOW THIS LINE                     ===
# ==============================================================================


def prepare_balanced_dataset(real_dirs, fake_dir, output_dir):
    """
    Creates a perfectly balanced training dataset by performing a two-pass
    sampling of the fake videos to exactly match the number of real videos.
    """
    # --- 1. Setup Output Directories ---
    train_real_path = os.path.join(output_dir, "train_real")
    train_fake_path = os.path.join(output_dir, "train_fake")

    print(f"Setting up output directories at: {output_dir}")
    os.makedirs(train_real_path, exist_ok=True)
    os.makedirs(train_fake_path, exist_ok=True)
    print("Done.")

    # --- 2. Process and Copy Real Videos ---
    print("\nProcessing real videos...")
    real_video_files = []
    for real_dir in real_dirs:
        if not os.path.isdir(real_dir):
            print(f"Warning: Real video directory not found, skipping: {real_dir}")
            continue
        print(f"  - Reading from: {real_dir}")
        for filename in os.listdir(real_dir):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                real_video_files.append(os.path.join(real_dir, filename))

    total_real_videos = len(real_video_files)
    if total_real_videos == 0:
        print("Error: No real videos found. Please check REAL_VIDEO_DIRS paths. Exiting.")
        return

    print(f"Found {total_real_videos} total real videos. This is our target number for fakes.")
    print("Copying real videos to train_real...")
    for src_path in tqdm(real_video_files, desc="Copying Real Videos"):
        shutil.copy2(src_path, train_real_path)
    print("Done.")

    # --- 3. Process and Sample Fake Videos ---
    print("\nProcessing fake videos...")
    if not os.path.isdir(fake_dir):
        print(f"Error: Fake video directory not found at {fake_dir}. Exiting.")
        return

    all_fake_videos = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    grouped_fakes = defaultdict(list)
    id_pattern = re.compile(r'^(id\d+)_')

    for filename in all_fake_videos:
        match = id_pattern.match(filename)
        if match:
            identity_id = match.group(1)
            grouped_fakes[identity_id].append(filename)

    num_unique_ids = len(grouped_fakes)
    if num_unique_ids == 0:
        print("Error: No fake videos with the 'idX_' pattern found. Exiting.")
        return
    
    print(f"Found {len(all_fake_videos)} total fake videos belonging to {num_unique_ids} unique identities.")

    # --- PASS 1: Fair Sampling by Identity ---
    print("\n--- Pass 1: Performing fair sampling across all identities ---")
    videos_per_id = total_real_videos // num_unique_ids
    remainder = total_real_videos % num_unique_ids
    print(f"Base quota per identity: {videos_per_id}. Remainder to distribute: {remainder}.")
    ids_with_extra_video = random.sample(list(grouped_fakes.keys()), remainder)

    selected_fake_videos = []
    already_selected_set = set()

    for identity_id, video_list in grouped_fakes.items():
        num_to_sample = videos_per_id + (1 if identity_id in ids_with_extra_video else 0)
        actual_sample_size = min(num_to_sample, len(video_list))
        sampled = random.sample(video_list, actual_sample_size)
        selected_fake_videos.extend(sampled)
        already_selected_set.update(sampled)
    
    print(f"Pass 1 resulted in {len(selected_fake_videos)} videos selected.")

    # --- PASS 2: Top-Up to Exact Number ---
    shortfall = total_real_videos - len(selected_fake_videos)
    if shortfall > 0:
        print(f"\n--- Pass 2: Shortfall of {shortfall} detected. Topping up... ---")
        
        # Create a pool of all available videos that haven't been selected yet
        supplemental_pool = [f for f in all_fake_videos if f not in already_selected_set]
        
        # Check if we have enough videos to fill the shortfall
        if len(supplemental_pool) < shortfall:
            print(f"Warning: Not enough unique fake videos available to meet the target.")
            print(f"Adding all {len(supplemental_pool)} remaining videos.")
            extra_videos = supplemental_pool
        else:
            # Randomly sample the exact number needed to fill the gap
            extra_videos = random.sample(supplemental_pool, shortfall)
            print(f"Randomly sampling {len(extra_videos)} extra videos from the remaining pool.")

        selected_fake_videos.extend(extra_videos)
    
    # --- 4. Copy Final Selection ---
    total_fake_to_copy = len(selected_fake_videos)
    print(f"\nTotal of {total_fake_to_copy} fake videos selected for final dataset.")
    print("Copying selected fake videos to train_fake...")

    for filename in tqdm(selected_fake_videos, desc="Copying Fake Videos"):
        src_path = os.path.join(FAKE_VIDEO_DIR, filename)
        dst_path = os.path.join(train_fake_path, filename)
        shutil.copy2(src_path, dst_path)
    print("Done.")

    # --- 5. Final Summary ---
    print("\n--- Dataset Preparation Summary ---")
    print(f"Total REAL videos copied: {len(os.listdir(train_real_path))}")
    print(f"Total FAKE videos copied: {len(os.listdir(train_fake_path))}")
    print(f"Balanced dataset created successfully in: {output_dir}")
    print("---------------------------------")


# --- Main execution block ---
if __name__ == "__main__":
    prepare_balanced_dataset(REAL_VIDEO_DIRS, FAKE_VIDEO_DIR, OUTPUT_DIR)