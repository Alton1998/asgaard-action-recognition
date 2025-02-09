from transformers import AutoProcessor, SiglipVisionModel
import torch
import argparse
import cv2
from PIL import Image
import itertools
from itertools import islice
import os
import csv
import time
from tqdm import tqdm
import pandas as pd


device = "cuda" if torch.cuda.is_available() else "cpu"

def video_to_pil_frames(video_path):
    print(f"Fetching Frames: {video_path}")
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total Frames: {total_frames}")

    for i in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        yield pil_image  # Yield one frame at a time

    video.release()


def batch_generator(iterable, batch_size):
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch
    
def generate_frame_vectors(model_id="google/siglip-base-patch16-224",video_path=""):
    print(f"Using checkpoint:{model_id}")
    model = SiglipVisionModel.from_pretrained(model_id,device_map=device)
    processor = AutoProcessor.from_pretrained(model_id)
    start_time = time.time()
    frames = video_to_pil_frames(video_path)
    print("Processing Frames")
    base_file_name_with_extension = os.path.basename(video_path)
    base_file_name, _ = os.path.splitext(base_file_name_with_extension)
    print("Generating Compressed Frame Vectors")
    for frame in tqdm(frames,desc="Generating Vectors"):
        inputs = processor(images=frame, return_tensors="pt").to(device)
        outputs = model(**inputs)
        for tensor in outputs.pooler_output:
            yield tensor.detach().flatten().reshape(1,-1)
def compute_cosine_similarity(frames):
    i=0
    try:
        next_frame = None
        current_frame = None
        while True:
            if next_frame is None and current_frame is None:
                current_frame = next(frames)
                next_frame = next(frames)
            similarity = torch.nn.functional.cosine_similarity(current_frame,next_frame).item()
            yield (i,similarity)
            current_frame = next_frame
            next_frame = next(frames)
            i = i + 1
    except StopIteration:
        print("End of frames Reached")
def generate_shots(cosine_similarities,threshold=0.9):
    for entry in tqdm(compute_cosine_similarity(frames),desc="Detecting Shots"):
        idx,similarities = entry
        if similarities < threshold:
            yield idx
def process_shot_to_tuples(shots):
    start = 0
    for shot in shots:
        entry = (start,shot)
        yield entry
        start = shot + 1
def write_to_csv(shots,output_path):
    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Start","End"])  # Header row
        for row in shots:  # Iterate over the generator
            writer.writerow(row)
def cut_video(video_path, frame_ranges, write_video=False):
    scene_list = []
    scene_list_seconds = []
    list_shot_boundary = []
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.path.dirname(video_path), base_name)
    os.makedirs(output_dir, exist_ok=True)
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    frame_rate = eval(video_streams[0]['r_frame_rate'])
    
    for i, (start_frame, end_frame) in enumerate(frame_ranges):
        start_time = start_frame / frame_rate
        start_time_seconds = convert_seconds(start_time)
        end_time = (end_frame + 1) / frame_rate
        end_time_seconds = convert_seconds(end_time)
        scene_list.append((start_time,end_time))
        scene_list_seconds.append((start_time_seconds,end_time_seconds))
        output_file = os.path.join(output_dir, f"{base_name}_part_{i+1}.mp4")
        if write_video:
            ffmpeg.input(video_path, ss=start_time, to=end_time).output(output_file).run()
    return scene_list,scene_list_seconds
def overlay_markers(video_path, shot_boundaries, output_path):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_path, exist_ok=True)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in tqdm(range(total_frames),desc="Writing Frames to Disk:"):
        ret, frame = cap.read()
        if not ret:
            break
        for start, end in shot_boundaries:
            if frame_idx == start or frame_idx == end:
                height, width, _ = frame.shape
                cv2.line(frame, (0, height//2), (width, height//2), (0, 0, 255), 5)
                cv2.putText(frame, "Shot Boundary", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imwrite(f"{output_path}/frame_{frame_idx}.jpg", frame)
    cap.release()
         
    
if __name__=="__main__":
    print(f"Device:{device}")
    parser = argparse.ArgumentParser(description="Shot Generator Using Cosine Similarity")
    parser.add_argument('--video', type=str, help="Video path")
    parser.add_argument('--threshold',type=float,help="Cosine Similarity Threshold")
    args = parser.parse_args()
    video_path = args.video
    threshold = args.threshold
    frames = generate_frame_vectors(video_path=video_path)
    similarities = compute_cosine_similarity(frames)
    shots = generate_shots(cosine_similarities=similarities,threshold=threshold)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    shot_tuples = list(process_shot_to_tuples(shots))  # Convert generator to list
    write_to_csv(shot_tuples, base_name+".csv")  # Now `shot_tuples` can be used again
    overlay_markers(video_path, shot_tuples, base_name)  # This will work as expected

    
        
    
    