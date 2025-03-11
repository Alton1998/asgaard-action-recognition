import os
import pandas as pd
import argparse
import ffmpeg
from PIL import Image
from tqdm import tqdm
import pickle

def extract_frames_to_pil(video_path, start_frame, end_frame):
    probe = ffmpeg.probe(video_path)
    fps = eval(next(stream for stream in probe["streams"] if stream["codec_type"] == "video")["r_frame_rate"])
    width = int(next(stream for stream in probe["streams"] if stream["codec_type"] == "video")["width"])
    height = int(next(stream for stream in probe["streams"] if stream["codec_type"] == "video")["height"])
    start_time = start_frame / fps
    end_time = end_frame / fps
    out, _ = (
        ffmpeg
        .input(video_path, ss=start_time, to=end_time)  # Trim video
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")  # Output raw RGB frames
        .run(capture_stdout=True, capture_stderr=True)  # Capture in memory
    )
    frame_size = width * height * 3
    frames = [
        Image.fromarray(np.frombuffer(out[i:i+frame_size], np.uint8).reshape((height, width, 3)), 'RGB')
        for i in range(0, len(out), frame_size)
    ]
    return frames
    
def process_data(video_path,data_file,shots_file,window=10,user_prompt="What is this?"):
    data_file_df = pd.read_csv(data_file)
    shots_file_df = pd.read_csv(shots_file)
    if len(data_file_df)!=len(shots_file_df):
        raise Exception("Data File and shots file should be of the same length")
    new_df = pd.concat([data_file_df,shots_file_df],axis=1)
    new_df["Action Formatted"] = new_df["Action"].fillna("No Action Found")
    formatted_conversation_data = []
    toks = "<image>" * (2*window)
    for _, row in tqdm(df.iterrows(),Creating Data Set):
        prompt = "<|im_start|>user"+ toks + f"\n{user_prompt}<|im_end|><|im_start|>assistant "+row["Action Formatted"] <|im_end|>"
        shot= row["Shot"]
        start =  0 if (shot-window) < 0 else (shot-window)
        end = shot+window
        frames = extract_frames_to_pil(video_path,start,end)
        data_dict = dict()
        data_dict["prompt"] = prompt
        data_dict["frames"] = frames
        formatted_conversation_data.append(data_dict)
    with open('parrot.pkl', 'wb') as f:
        pickle.dump(formatted_conversation_data, f)
        
        
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Video DataSet Preparator")
    parser.add_argument('--video', type=str, help="Video path")
    parser.add_argument('--data', type=str, help="CSV data path")
    parser.add_argument('--shots',type=str, help="Shots CSV data path")
    parser.add_argument('--window',type=int, help="Number of frames you wanna annotate")
    parser.add_argument('--prompt',type=str, help="User  Prompt")
    args = parser.parse_args()
    video_path = args.video
    data_path = args.data
    shots_path = args.shots
    window = args.window
    prompt = args.prompt
    process_data(video_path,data_path,shots_path,window,prompt)
    
    
    