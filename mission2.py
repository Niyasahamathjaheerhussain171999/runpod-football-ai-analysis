import torch
from ultralytics import YOLO
import cv2
import easyocr
import csv
import yt_dlp

# --- 1. Setup ---
# Select device
if torch.cuda.is_available():
    device_to_use = 0
    print(f"GPU is available! Device: {torch.cuda.get_device_name(0)}")
else:
    device_to_use = 'cpu'
    print("GPU not available, running on CPU.")

# --- Video Download Section ---
video_url = 'https://youtube.com/shorts/MbY6bTCA0ww?feature=shared'
video_path = 'football_clip.mp4'
output_csv_path = 'tracking_results.csv'

ydl_opts = {'format': 'best[ext=mp4]', 'outtmpl': video_path, 'quiet': True}

print(f"\nDownloading video from {video_url}...")
try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    print(f"Video saved as {video_path}")
except Exception as e:
    print(f"Error downloading video: {e}")
    exit()

# Load the models
yolo_model = YOLO('yolov8n.pt')
ocr_reader = easyocr.Reader(['en'])

# --- 2. Video Processing ---
# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Prepare to store results
all_results = []
frame_number = 0
print("\nStarting video processing for Mission 2...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Use the 'track' method on the current frame
    results = yolo_model.track(frame, persist=True, device=device_to_use, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            player_crop = frame[y1:y2, x1:x2]
            ocr_results = ocr_reader.readtext(player_crop, detail=0)
            jersey_guess = ""
            for text in ocr_results:
                if text.isdigit():
                    jersey_guess = text
                    break

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            all_results.append([frame_number, track_id, jersey_guess, center_x, center_y])
    frame_number += 1

cap.release()

print("Video processing complete. Saving results to CSV...")
with open(output_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'track_id', 'jersey_guess', 'x', 'y'])
    writer.writerows(all_results)

print(f"Mission 2 Complete! Data saved to {output_csv_path}")