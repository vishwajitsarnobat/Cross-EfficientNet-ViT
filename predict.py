# predict.py

import os
import argparse
import json
import yaml

import torch
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from termcolor import cprint

import config
from preprocessing.detect_faces import run_face_detection
from preprocessing.extract_crops import run_crop_extraction
from deepfakes_dataset import DeepFakesDataset
from cross_efficient_vit import CrossEfficientViT
from utils import custom_video_round

def create_visual_report(video_path, frame_preds, final_verdict, confidence, fps=30.0, use_time_axis=False):
    """
    Generates a clean, beautiful, and clutterless PNG report containing ONLY the
    verdict, confidence, and a gradient-filled probability graph.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = os.path.join(config.results_path, f"{base_name}_prediction_report.png")
    
    frame_preds = np.array(frame_preds)
    num_frames = len(frame_preds)
    fake_threshold = 0.5

    # --- Setup X-Axis ---
    if use_time_axis and fps > 0:
        x_axis = np.arange(num_frames) / fps
        x_label = "Time (seconds)"
    else:
        x_axis = np.arange(num_frames)
        x_label = "Frame Number"

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax_verdict, ax_plot) = plt.subplots(
        2, 1, figsize=(16, 9), dpi=120, 
        gridspec_kw={'height_ratios': [1, 5]} # Top panel is smaller
    )
    fig.patch.set_facecolor('#f8f9fa')

    # --- 1. Top Panel: Verdict and Confidence ---
    ax_verdict.set_facecolor('#f8f9fa')
    ax_verdict.axis('off')
    verdict_color = '#d9534f' if final_verdict == "FAKE" else '#5cb85c'
    
    ax_verdict.text(0.5, 0.65, f"VERDICT: {final_verdict}", ha='center', va='center', 
                    fontsize=32, fontweight='bold', color=verdict_color)
    ax_verdict.text(0.5, 0.25, f"Confidence: {confidence:.2%}", ha='center', va='center', 
                    fontsize=20, color='#343a40')

    # --- 2. Bottom Panel: The Graph ---
    ax_plot.set_facecolor('#ffffff')
    ax_plot.plot(x_axis, frame_preds, color='#007bff', linewidth=2.5, label='Fake Probability')
    ax_plot.axhline(y=fake_threshold, color='#343a40', linestyle='--', linewidth=1.5, alpha=0.7, 
                    label=f'Fake Threshold ({fake_threshold})')
    
    ax_plot.fill_between(x_axis, fake_threshold, frame_preds, where=frame_preds >= fake_threshold, 
                         facecolor='#d9534f', interpolate=True, alpha=0.4)
    ax_plot.fill_between(x_axis, fake_threshold, frame_preds, where=frame_preds < fake_threshold, 
                         facecolor='#5cb85c', interpolate=True, alpha=0.4)

    ax_plot.set_title('Frame-by-Frame Fake Probability Analysis', fontsize=18, fontweight='bold', pad=20)
    ax_plot.set_xlabel(x_label, fontsize=14)
    ax_plot.set_ylabel('Probability Score', fontsize=14)
    ax_plot.set_ylim(0, 1)
    if num_frames > 1:
        ax_plot.set_xlim(0, x_axis[-1])
    
    ax_plot.tick_params(axis='both', which='major', labelsize=12)
    ax_plot.spines['top'].set_visible(False)
    ax_plot.spines['right'].set_visible(False)
    
    ax_plot.legend(fontsize=12)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(output_filename)
    plt.close()
    return output_filename

def save_json_report(video_path, verdict, confidence, frame_preds, fps, total_frames):
    """Saves a detailed JSON report."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = os.path.join(config.results_path, f"{base_name}_prediction_report.json")
    duration = total_frames / fps if fps > 0 else 0

    report = {
        'video_information': {
            'filename': os.path.basename(video_path),
            'path': video_path,
            'duration_seconds': round(duration, 2),
            'total_frames': total_frames,
            'frames_with_faces': len(frame_preds)
        },
        'prediction_summary': {
            'verdict': verdict,
            'confidence': round(confidence, 4)
        },
        'frame_by_frame_results': {
            'fake_probabilities': [round(float(p), 4) for p in frame_preds]
        }
    }

    with open(output_filename, 'w') as f:
        json.dump(report, f, indent=4)
    return output_filename

def main():
    parser = argparse.ArgumentParser(description="Predict if a video is a deepfake.")
    parser.add_argument("video_path", type=str, help="Path to the video file to be analyzed.")
    parser.add_argument("--model_path", type=str, default=config.test_model_path, 
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--time_axis", action="store_true", 
                        help="Use time in seconds for the x-axis of the graph instead of frame numbers.")
    opt = parser.parse_args()

    if not os.path.exists(opt.video_path):
        cprint(f"Error: Video file not found at '{opt.video_path}'", 'red')
        return

    # Preprocessing
    run_face_detection([opt.video_path])
    run_crop_extraction([opt.video_path])

    # Setup
    os.makedirs(config.results_path, exist_ok=True)
    with open(config.architecture_config_path, 'r') as ymlfile:
        arch_config = yaml.safe_load(ymlfile)
    image_size = arch_config['model']['image-size']

    transform = DeepFakesDataset(frame_label_list=[], image_size=image_size, mode='validation').create_val_transform(image_size)

    # Load Model
    model = CrossEfficientViT(config=arch_config)
    if not os.path.exists(opt.model_path):
        cprint(f"Error: Model file not found at '{opt.model_path}'", 'red')
        return
    model.load_state_dict(torch.load(opt.model_path, map_location=config.device))
    model.eval().to(config.device)

    # Get video info for reports
    cap = cv2.VideoCapture(opt.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Load Frames
    video_id = os.path.splitext(os.path.basename(opt.video_path))[0]
    crops_dir = os.path.join(config.output_path, "crops", video_id)
    
    if not os.path.exists(crops_dir) or not os.listdir(crops_dir):
        cprint(f"Error: No faces were detected in this video. Cannot predict.", 'red')
        return

    frames = []
    for frame_file in sorted(os.listdir(crops_dir)):
        image = cv2.imread(os.path.join(crops_dir, frame_file))
        if image is None: continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(transform(image=image_rgb)['image'])
    
    if not frames:
        cprint(f"Error: Failed to load valid frames from '{crops_dir}'.", 'red')
        return
        
    frames_tensor = torch.stack(frames).to(config.device)

    # Predict
    frame_preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(frames_tensor), config.test_batch_size), desc="Analyzing Frames"):
            batch = frames_tensor[i : i + config.test_batch_size]
            pred = torch.sigmoid(model(batch))
            frame_preds.extend(pred.cpu().numpy().flatten().tolist())

    # Process results
    video_level_pred_score = custom_video_round(frame_preds)
    final_verdict = "FAKE" if video_level_pred_score > 0.5 else "REAL"
    confidence = video_level_pred_score if final_verdict == "FAKE" else 1 - video_level_pred_score

    # --- Final Console Output ---
    verdict_color = 'red' if final_verdict == "FAKE" else 'green'
    cprint(f"\nVerdict: {final_verdict}", verdict_color, attrs=['bold'])
    cprint(f"Confidence: {confidence:.2%}\n", verdict_color)

    # --- Generate and Save Reports ---
    report_path = create_visual_report(opt.video_path, frame_preds, final_verdict, confidence, fps, opt.time_axis)
    json_path = save_json_report(opt.video_path, final_verdict, confidence, frame_preds, fps, total_frames)
    
    cprint(f"✓ Visual report saved to: {report_path}", 'cyan')
    cprint(f"✓ JSON data saved to:   {json_path}", 'cyan')

if __name__ == "__main__":
    main()