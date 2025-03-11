import pandas as pd
import argparse
import ffmpeg
import os
from tqdm import tqdm

def write_clips(shots_file, video_file):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(base_name, exist_ok=True)
    df = pd.read_csv(shots_file)
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    frame_rate = eval(video_streams[0]['r_frame_rate'])
    for index, row in tqdm(df.iterrows(),desc="Writing clips"):
        start_time = row["Start"] / frame_rate
        end_time = row["End"] / frame_rate
        output_file = os.path.join(base_name, f"{base_name}_part_{index}.mp4")
        try:
            ffmpeg.input(video_path, ss=start_time, to=end_time, hwaccel="cuda").output(output_file).run()
        except Exception as e:
            print(f"An Expetion occured skipping clip {index}")
            continue

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Clip writer")
    parser.add_argument('--video', type=str, help="Video path")
    parser.add_argument('--shots',type=str,help="Shots file path")
    args = parser.parse_args()
    video_path = args.video
    shots_path = args.shots
    write_clips(shots_path,video_path)