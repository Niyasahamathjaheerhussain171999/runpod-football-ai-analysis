# Import the necessary libraries
from ultralytics import YOLO
import yt_dlp
import torch

# --- 1. Check for GPU and Set the Device ---
# This is the key change. We create a variable to hold our choice.
if torch.cuda.is_available():
    device_to_use = 0 # Use the first GPU
    print(f"GPU is available! Device: {torch.cuda.get_device_name(0)}")
else:
    device_to_use = 'cpu' # Use the CPU
    print("GPU not available, running on CPU.")

# --- 2. Download a Sample Video ---
# This part is unchanged.
video_url = 'https://youtube.com/shorts/MbY6bTCA0ww?feature=shared' # A 1-min clip of football skills
output_filename = 'football_clip.mp4'

# Download options
ydl_opts = {
    'format': 'best[ext=mp4]',
    'outtmpl': output_filename,
    'quiet': True,
}

print(f"\nDownloading video from {video_url}...")
try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    print(f"Video saved as {output_filename}")
except Exception as e:
    print(f"Error downloading video: {e}")
    exit()

# --- 3. Run the AI Model ---
print("\nLoading YOLOv8 model...")
model = YOLO('yolov8n.pt')

print("Running player and ball detection on the video...")
# This is the second key change. We use our new variable here.
results = model.predict(source=output_filename, save=True, device=device_to_use)

print("\nMission 1 Complete!")
print("Your result video has been saved in a 'runs/detect/' folder.")