uv run preprocessing/detect_faces.py --processes 8
uv run preprocessing/extract_crops.py
uv run train.py --num_epochs 50 --patience 7
python test.py --model_path models/efficientnet_checkpoint25.pth
python predict.py --video_path /path/to/your/new_video.mp4

shuf -n 200 -e ./Celeb-synthesis/* | xargs -I{} cp "{}" ./test-fake/