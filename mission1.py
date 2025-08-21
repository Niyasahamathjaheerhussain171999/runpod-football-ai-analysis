# Import the necessary libraries
from ultralytics import YOLO
import yt_dlp
import torch

# --- 1. Check for GPU ---
# This is a good practice to confirm you're using the GPU when on RunPod.
if torch.cuda.is_available():
    print(f"GPU is available! Device: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available, running on CPU.")

# --- 2. Download a Sample Video ---
# We use a library called yt-dlp to download a video directly.
# This is much faster than uploading.
video_url = 'https://youtu.be/ca-e5EuaYgU' # A short, 1-min football highlights clip
output_filename = 'football_clip.mp4'

# Download options
ydl_opts = {
    'format': 'best[ext=mp4]', # Get the best quality mp4
    'outtmpl': output_filename, # Save it with our desired filename
    'quiet': True, # Don't print too much download info
}

print(f"\nDownloading video from {video_url}...")
try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    print(f"Video saved as {output_filename}")
except Exception as e:
    print(f"Error downloading video: {e}")
    exit() # Stop if download fails

# --- 3. Run the AI Model ---
# This is the core of the mission.
print("\nLoading YOLOv8 model...")
model = YOLO('yolov8n.pt')  # 'n' is the smallest, fastest model - good for testing

print("Running player and ball detection on the video...")
# The 'predict' function does all the heavy work on the GPU.
# 'save=True' tells it to save the output video with the boxes drawn on it.
results = model.predict(source=output_filename, save=True, device=0) # device=0 means use the first GPU

print("\nMission 1 Complete!")
print("Your result video has been saved in a 'runs/detect/' folder.")