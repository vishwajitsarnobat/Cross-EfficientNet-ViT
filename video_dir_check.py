import cv2
import os
import argparse
import datetime

def analyze_videos(video_directory):
    """
    Analyzes all video files in a given directory to determine their
    duration and total number of frames.

    Saves the information to a text file named after the directory
    (e.g., 'my_videos.txt') in the current working directory, and
    prints the totals to the console.
    """
    total_frames_all_videos = 0
    total_duration_all_videos = 0.0

    # Get the base name of the input directory to use as the filename.
    # os.path.normpath handles potential trailing slashes (e.g., "videos/" vs "videos").
    dir_name = os.path.basename(os.path.normpath(video_directory))
    output_file = f"{dir_name}.txt"

    video_files = [f for f in os.listdir(video_directory) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    # By default, open() will create the file in the current working directory.
    with open(output_file, 'w') as f:
        f.write(f"Video Analysis Report for Directory: {video_directory}\n")
        f.write("="*50 + "\n\n")

        for video_file in video_files:
            video_path = os.path.join(video_directory, video_file)

            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error opening video file: {video_file}")
                    continue

                # Get video properties
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Calculate duration
                if fps > 0:
                    duration = frame_count / fps
                else:
                    duration = 0

                # Update totals
                total_frames_all_videos += frame_count
                total_duration_all_videos += duration

                # Write to file
                f.write(f"File: {video_file}\n")
                f.write(f"  - Duration: {str(datetime.timedelta(seconds=int(duration)))}\n")
                f.write(f"  - Total Frames: {frame_count}\n\n")

                cap.release()

            except Exception as e:
                print(f"Could not process {video_file}. Error: {e}")

    print(f"\nAnalysis complete. Results saved to '{output_file}' in the current directory.")

    print("\n--- Totals for all videos ---")
    print(f"Total Running Time: {str(datetime.timedelta(seconds=int(total_duration_all_videos)))}")
    print(f"Total Frames: {total_frames_all_videos}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze videos in a directory.")
    parser.add_argument("directory", type=str, help="The directory containing the video files.")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found at {args.directory}")
    else:
        analyze_videos(args.directory)