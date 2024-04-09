# use this script to extract frames from video clips and save them in a folder
# Usage: python video_clip_processor.py output_folder
import os
import json
import torch
import numpy as np
import pickle

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


from lavis.datasets.data_utils import load_pickle
import re
import cv2
from tqdm import tqdm
from moviepy.editor import VideoFileClip



def extract_frames(video_path, output_folder, num_frames):
    os.makedirs(output_folder, exist_ok=True)
    video = VideoFileClip(video_path)
    total_frames = video.reader.nframes
    downsample_factor = max(1, total_frames // num_frames)

    for i in range(0, total_frames, downsample_factor):
        frame = video.get_frame(i / video.fps)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        output_path = os.path.join(output_folder, f"frame_{i}.jpg")
        image.save(output_path)

    video.reader.close()

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def main():

    data_path = "data/ego4d/v2/nlq_official_v2_omnivore_video_fp16_official_128_bert.pkl"
    data = load_pickle(data_path)

    train_data = data["val_set"]

    narration_file_path = "data/ego4d/v2/annotations/narration.json"
    narration_data = load_json_file(narration_file_path)

    nlq_file_path = "data/ego4d/v2/annotations/nlq_val.json"
    nlq_data = load_json_file(nlq_file_path)

    video_clip_dict = {}
    clip_id_list = []

    for video in nlq_data["videos"]:
        video_uid = video['video_uid']
        for clip in video['clips']:
            clip_uid = clip['clip_uid']
            clip_s_time = clip['video_start_sec']
            clip_e_time = clip['video_end_sec']
            video_clip_dict[clip_uid] = [video_uid, clip_s_time, clip_e_time]


    for i in tqdm(range(len(train_data))):
        record = train_data[i]
        clip_id, start_time, end_time = record["vid"], record["s_time"], record["e_time"]
        s_ind, e_ind = record["s_ind"], record["e_ind"]
        question = record["query"]
        video_id = video_clip_dict[clip_id][0]
        clip_s_time, clip_e_time = video_clip_dict[clip_id][1], video_clip_dict[clip_id][2]
        video_clip = narration_data[video_id]
        if clip_id in clip_id_list:
            continue
        else:
            clip_id_list.append(clip_id)
        narrations = []

        video_path = os.path.join('data/ego4d/v2/clips', (clip_id+'.mp4'))
        
        if not os.path.exists(video_path):
            continue
        output_folder = os.path.join(sys.argv[1], clip_id)
        os.makedirs(output_folder, exist_ok=True)

        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        

        if "narration_pass_1" in video_clip:
            narrations += video_clip["narration_pass_1"]["narrations"]
        if "narration_pass_2" in video_clip:
            narrations += video_clip["narration_pass_2"]["narrations"]
        for narration in narrations:
            timestamp = narration['timestamp_sec']-clip_s_time
            timestamp_frame = narration['timestamp_frame']
            # assert timestamp>=0
            if timestamp>=0 and timestamp<=480:
                target_time = timestamp
                target_frame_index = int(target_time * fps)
                video.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
                success, frame = video.read()
                output_path = os.path.join(output_folder, f"frame_{timestamp_frame}_[{narration['narration_text'][3:].replace(' ','_')}].jpg")
                if success:
                    cv2.imwrite(output_path, frame)
                else:
                    pass
        video.release()

if __name__ == "__main__":
    main()
