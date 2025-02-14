import cv2
from PIL import Image
import os
import argparse
from tqdm import tqdm
def video_to_pil_frames(video_path,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Fetching Frames: {video_path}")
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total Frames: {total_frames}")

    for i in tqdm(range(total_frames),desc="Writing Frames"):
        ret, frame = video.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        pil_image.save(f"{output_dir}/frame_{i}.png")

    video.release()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Shot Generator Using Cosine Similarity")
    parser.add_argument('--video', type=str, help="Video path")
    args = parser.parse_args()
    video_path = args.video
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_to_pil_frames(video_path=video_path,output_dir=base_name)