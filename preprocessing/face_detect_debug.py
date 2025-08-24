# debug_cropping.py
import cv2
import json
import os
import argparse

def debug_single_frame(video_path, json_path, frame_number=10):
    """
    This function draws the bounding boxes on a specific frame to let us see
    what the cropping script is doing.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    # Load the bounding box data
    with open(json_path, 'r') as f:
        bboxes_dict = json.load(f)

    # Go to the specified frame in the video
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = capture.read()
    capture.release()

    if not success:
        print(f"Error: Could not read frame {frame_number} from the video.")
        return

    # Get the bounding boxes for this frame
    bboxes = bboxes_dict.get(str(frame_number))
    if not bboxes:
        print(f"No faces detected in frame {frame_number} according to the JSON file.")
        return

    print(f"Found {len(bboxes)} face(s) in frame {frame_number}.")

    for bbox in bboxes:
        # --- This is the same logic from extract_crops.py ---
        
        # Step 1: Scale the coordinates
        xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]

        # Draw the raw scaled rectangle (in BLUE)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(frame, 'Scaled Box', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Step 2: Calculate square padding
        w = xmax - xmin
        h = ymax - ymin
        p_h, p_w = 0, 0
        if h > w: p_w = (h - w) // 2
        elif w > h: p_h = (w - h) // 2

        # Step 3: Define the final crop area (in GREEN)
        crop_xmin = max(xmin - p_w, 0)
        crop_ymin = max(ymin - p_h, 0)
        crop_xmax = xmax + p_w
        crop_ymax = ymax + p_h
        cv2.rectangle(frame, (crop_xmin, crop_ymin), (crop_xmax, crop_ymax), (0, 255, 0), 2)
        cv2.putText(frame, 'Final Crop Area', (crop_xmin, crop_ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the output image
    output_filename = "debug_output.jpg"
    cv2.imwrite(output_filename, frame)
    print(f"\nSaved debug image to '{output_filename}'.")
    print(" - The BLUE box is the direct output from the face detector (scaled).")
    print(" - The GREEN box is the final square area that will be cropped.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Debug the face cropping process.")
    parser.add_argument('--video', required=True, help="Path to the video file.")
    parser.add_argument('--json', required=True, help="Path to the corresponding .json bounding box file.")
    parser.add_argument('--frame', type=int, default=10, help="The frame number to inspect.")
    args = parser.parse_args()

    debug_single_frame(args.video, args.json, args.frame)