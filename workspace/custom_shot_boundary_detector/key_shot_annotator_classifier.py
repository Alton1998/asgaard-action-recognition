import pandas as pd
import cv2
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
import argparse
from tqdm import tqdm
import os


device = "cuda" if torch.cuda.is_available() else "cpu"

def annotate_key_shots(key_shots,video_path,model_id="llava-hf/llava-interleave-qwen-0.5b-hf",user_prompt = "What is this scene about?"):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16,load_in_4bit=True)
    model.to("cuda")
    toks = "<image>"
    prompt = "<|im_start|>user"+ toks + f"\n{user_prompt}<|im_end|><|im_start|>assistant"
    for i in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break
        if i in key_shots:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(model.device, model.dtype)
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            output_decoded = processor.decode(output[0][2:], skip_special_tokens=True)[len(user_prompt)+10:]
            yield (pil_image,output_decoded)

def write_to_sheets(title,image_list,output_path):
    wb = Workbook()
    ws = wb.active
    ws.title = title
    ws.append(["Image", "Description"])
    for i, (img, desc) in tqdm(enumerate(image_list, start=2),desc="Annotating"): 
        img_path = f"image_{i-1}.png"
        img.save(img_path)  

        ws[f"B{i}"] = desc
        excel_img = XLImage(img_path)
        ws.add_image(excel_img, f"A{i}")
    wb.save(output_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Key Shot Annotation")
    parser.add_argument('--csv', type=str, help="CSV Path")
    parser.add_argument('--video', type=str, help="Video Path")
    args = parser.parse_args()
    video_path = args.video
    csv_path = args.csv
    key_shots = pd.read_csv("1_shots.csv").values.reshape(1,-1)[0].tolist()
    image_list = annotate_key_shots(key_shots=key_shots,video_path=video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    write_to_sheets(base_name,image_list,base_name+".xlsx")
    



