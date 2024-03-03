"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import torch
import numpy as np
import pickle

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from lavis.datasets.data_utils import load_video_features, pad_video_seq, pad_seq, pad_char_seq, load_pickle
from nuscenes.nuscenes import NuScenes
import re
from collections import OrderedDict
import tqdm
import random

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_img_frames(clip_id, timestamp_frame):
    img_dir = "data/ego4d/output_data_narra"
    img_path = os.path.join(img_dir, clip_id)
    file_pattern = f"frame_{timestamp_frame}_"

    for root, dirnames, filenames in os.walk(img_path):
        for filename in filenames:
            if file_pattern in filename:
                full_path = os.path.join(root, filename)
                return full_path
    return None


data_infos = []
questions = []
answers = []
time = []

def get_ego4d():
    data_path = "data/ego4d/v2/nlq_official_v2_omnivore_video_fp16_official_128_bert.pkl"
    data = load_pickle(data_path)
    train_data = data["train_set"]
    val_data = data["val_set"]
    train_data.extend(val_data)

    # process answers
    narration_file_path = "data/ego4d/v2/annotations/narration.json"
    narration_data = load_json_file(narration_file_path)

    nlq_file_path1 = "data/ego4d/v2/annotations/nlq_train.json"
    nlq_data = load_json_file(nlq_file_path1)

    nlq_file_path2 = "data/ego4d/v2/annotations/nlq_val.json"
    nlq_data2 = load_json_file(nlq_file_path2)
    nlq_data["videos"].extend(nlq_data2["videos"])

    video_clip_dict = {}
    clip_id_list = []

    for video in nlq_data["videos"]:
        video_uid = video['video_uid']
        for clip in video['clips']:
            clip_uid = clip['clip_uid']
            clip_s_time = clip['video_start_sec']
            clip_e_time = clip['video_end_sec']
            video_clip_dict[clip_uid] = [video_uid, clip_s_time, clip_e_time]
    

    for i in tqdm.tqdm(range(len(train_data))):
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
        if "narration_pass_1" in video_clip:
            narrations = video_clip["narration_pass_1"]["narrations"]
        else:
            if "narration_pass_2" in video_clip:
                narrations = video_clip["narration_pass_2"]["narrations"]
            else:
                continue
        narrations = sorted(narrations, key=lambda x: x['timestamp_sec'])
        
        # memory query is here:
        mem_length = 20
        time_interval = 5
        his_threshold = 10
        for i in range(0, len(narrations)-mem_length, time_interval):
            narration_tmp = narrations[i:i+mem_length]
            answer = []
            img_path = []
            time = []
            if narration_tmp[-1]['timestamp_sec']<narration_tmp[0]['timestamp_sec']:
                continue
            for narration in narration_tmp:
                timestamp = narration['timestamp_sec']-clip_s_time
                timestamp_frame = narration['timestamp_frame']
                if timestamp>=0 and timestamp<=480:
                    img_file = get_img_frames(clip_id, timestamp_frame)
                    if img_file is not None:
                        img_path.append(img_file)
                        answer.append(narration['narration_text'][3:])
                        time.append(timestamp)
            if len(img_path) == mem_length:
                start_index = random.randrange(len(answer) - 2 - his_threshold)
                narr = answer[start_index:start_index+3]
                question = f"What happened between '{narr[0]}' and '{narr[2]}'?"
                answer = narr[1]
                time = [0 for _ in time]
                questions.append(question)
                answers.append([answer])
                data_infos.append(img_path)
                time.append([t-time[-1] for t in time])

        # memory timestamp query is here:
        for i in range(0, len(narrations)-mem_length, time_interval):
            narration_tmp = narrations[i:i+mem_length]
            answer = []
            img_path = []
            time = []
            if narration_tmp[-1]['timestamp_sec']<narration_tmp[0]['timestamp_sec']:
                continue
            for narration in narration_tmp:
                timestamp = narration['timestamp_sec']-clip_s_time
                timestamp_frame = narration['timestamp_frame']
                if timestamp>=0 and timestamp<=480:
                    img_file = get_img_frames(clip_id, timestamp_frame)
                    if img_file is not None:
                        img_path.append(img_file)
                        answer.append(narration['narration_text'][3:])
                        time.append(timestamp)
            if len(img_path) == mem_length:
                start_index = random.randrange(len(answer) - 1- his_threshold)
                narr = answer[start_index]
                his_time = abs(round(time[start_index]-time[-1], 1))
                question = f"What happened {his_time} seconds before in the history?"
                answer = narr
                questions.append(question)
                answers.append([answer])
                data_infos.append(img_path)
                time.append([t-time[-1] for t in time])

        # future timestamp query is here:
        mem_length += 1
        for i in range(0, len(narrations)-mem_length, time_interval):
            narration_tmp = narrations[i:i+mem_length]
            answer = []
            img_path = []
            time = []
            if narration_tmp[-1]['timestamp_sec']<narration_tmp[0]['timestamp_sec']:
                continue
            for narration in narration_tmp:
                timestamp = narration['timestamp_sec']-clip_s_time
                timestamp_frame = narration['timestamp_frame']
                if timestamp>=0 and timestamp<=480:
                    img_file = get_img_frames(clip_id, timestamp_frame)
                    if img_file is not None:
                        img_path.append(img_file)
                        answer.append(narration['narration_text'][3:])
                        time.append(timestamp)
            if len(img_path) == mem_length:
                fu_time = abs(round(time[-1]-time[-2], 1))
                question = f"What will happen in the next {fu_time} seconds in the future?"
                answer = answer[-1]
                time = time[:-1]
                questions.append(question)
                answers.append([answer])
                data_infos.append(img_path[:mem_length-1])
                time.append([t-time[-1] for t in time])


def get_drivelm_narration():
    annotation = json.load(open("data/drivelm/train.json", "r"))
    annotation1 = json.load(open("data/drivelm/val.json", "r"))
    annotation.update(annotation1)

    data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
    for idx, info in enumerate(data_info):
        scene_token = info['scene_token']
        timestamp = info['cams']['CAM_FRONT']['timestamp']
        image_path = info['cams']["CAM_FRONT"]['data_path']

        if scene_token not in annotation:
            continue
        value = annotation[scene_token]
        # scene_description = value['scene_description']
        scene_key_frame = value['key_frame']
        frame_id = str(timestamp)
        if frame_id in scene_key_frame:

            value1 = scene_key_frame[frame_id]

            if "Perception" in value1:
                Perception_q = value1['Perception']['q']
                Perception_a = value1['Perception']['a']
            else:
                Perception_q = []
                Perception_a = []

            if "Prediction and Planning" in value1:
                Prediction_q = value1['Prediction and Planning']['q']
                Prediction_a = value1['Prediction and Planning']['a']
            else:
                Prediction_q = []
                Prediction_a = []
                
            Question = Perception_q# + Prediction_q
            Answer = Perception_a# + Prediction_a

            assert len(Question) == len(Answer)

            for idx in range(len(Question)):
                questions.append(Question[idx])
                answers.append([Answer[idx]])


def get_boxqa():
    annotation = json.load(open("data/embodied/BOXQA_train_v3.json", "r"))
    annotation1 = json.load(open("data/embodied/BOXQA_val_v3.json", "r"))
    annotation.update(annotation1)

    data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
    for idx, info in enumerate(data_info):
        scene_token = info['scene_token']
        timestamp = info['cams']['CAM_FRONT']['timestamp']
        image_path = info['cams']["CAM_FRONT"]['data_path']

        if scene_token not in annotation:
            continue
        value = annotation[scene_token]
        # scene_description = value['scene_description']
        scene_key_frame = value['key_frame']
        frame_id = str(timestamp)
        if frame_id in scene_key_frame:
            value1 = scene_key_frame[frame_id]

            if value1 is None:
                continue

            if "BOX QA" in value1:
                BOX_q = value1['BOX QA']['q']
                BOX_a = value1['BOX QA']['a']
                assert len(BOX_q) == len(BOX_a)
            else:
                BOX_q = []
                BOX_a = []
                
            Question = BOX_q
            Answer = BOX_a

            assert len(Question) == len(Answer)

            for idx in range(len(Question)):
                # reduce the decimals of questions
                coor1, coor2 = Question[idx].split('<c, CAM_FRONT, ')[-1].split('>')[0].split(', ')
                num1, num2 = round(float(coor1), 1), round(float(coor2), 1)
                Question[idx] = Question[idx].replace(coor1, str(num1))
                Question[idx] = Question[idx].replace(coor2, str(num2))
                Question[idx] += ", then describe the class of this object."

                questions.append(Question[idx])


def get_future_boxqa_no_word():
    annotation = json.load(open("data/embodied/Future_BOXQA_train.json", "r"))
    annotation1 = json.load(open("data/embodied/Future_BOXQA_val.json", "r"))
    annotation.update(annotation1)

    data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
    for idx, info in enumerate(data_info):
        scene_token = info['scene_token']
        timestamp = info['cams']['CAM_FRONT']['timestamp']

        if scene_token not in annotation:
            continue
        value = annotation[scene_token]
        # scene_description = value['scene_description']
        scene_key_frame = value['key_frame']
        frame_id = str(timestamp)
        if frame_id in scene_key_frame:
            value1 = scene_key_frame[frame_id]

            if value1 is None:
                continue

            if "Future BOX QA" in value1:
                Future_BOX_q = value1['Future BOX QA']['q']
                Future_BOX_a = value1['Future BOX QA']['a']
                assert len(Future_BOX_q) == len(Future_BOX_a)
            else:
                Future_BOX_q = []
                Future_BOX_a = []

            Question = Future_BOX_q
            Answer = Future_BOX_a

            assert len(Question) == len(Answer)

            for idx in range(len(Question)):
                # reduce the decimals of questions
                coor1, coor2 = Question[idx].split('<c, CAM_FRONT, ')[-1].split('>')[0].split(', ')
                num1, num2 = round(float(coor1), 1), round(float(coor2), 1)
                Question[idx] = Question[idx].replace(coor1, str(num1))
                Question[idx] = Question[idx].replace(coor2, str(num2))
                Question[idx] += ", then describe the class of this object."

                x, y, z, label = Answer[idx].split(', ')
                x = x.split('A: ')[-1]
                if round(float(y)) > 70:
                    continue
                index1 = f"{round(float(x), 1)}"
                index2 = f"{round(float(y), 1)}"
                index3 = f"{round(float(z), 1)}"

                vocab = [index1, index2, index3]
                strings = 'Location: '+' '.join(vocab) + ' Label: ' + label

                questions.append(Question[idx])


def get_history_boxqa_no_word():
    annotation = json.load(open("data/embodied/Hist_BOXQA_train.json", "r"))
    annotation1 = json.load(open("data/embodied/Hist_BOXQA_val.json", "r"))
    annotation.update(annotation1)

    data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
    for idx, info in enumerate(data_info):
        scene_token = info['scene_token']
        timestamp = info['cams']['CAM_FRONT']['timestamp']

        if scene_token not in annotation:
            continue
        value = annotation[scene_token]
        # scene_description = value['scene_description']
        scene_key_frame = value['key_frame']
        frame_id = str(timestamp)
        if frame_id in scene_key_frame:
            value1 = scene_key_frame[frame_id]

            if value1 is None:
                continue

            if "History BOX QA" in value1:
                Future_BOX_q = value1['History BOX QA']['q']
                Future_BOX_a = value1['History BOX QA']['a']
                assert len(Future_BOX_q) == len(Future_BOX_a)
            else:
                Future_BOX_q = []
                Future_BOX_a = []

            Question = Future_BOX_q
            Answer = Future_BOX_a

            assert len(Question) == len(Answer)

            for idx in range(len(Question)):
                # reduce the decimals of questions
                coor1, coor2 = Question[idx].split('<c, CAM_FRONT, ')[-1].split('>')[0].split(', ')
                num1, num2 = round(float(coor1), 1), round(float(coor2), 1)
                Question[idx] = Question[idx].replace(coor1, str(num1))
                Question[idx] = Question[idx].replace(coor2, str(num2))
                Question[idx] += ", then describe the class of this object."

                x, y, z, label = Answer[idx].split(', ')
                x = x.split('A: ')[-1]
                if round(float(y)) > 70:
                    continue
                index1 = f"{round(float(x), 1)}"
                index2 = f"{round(float(y), 1)}"
                index3 = f"{round(float(z), 1)}"

                vocab = [index1, index2, index3]
                strings = 'Location: '+' '.join(vocab) + ' Label: ' + label

                questions.append(Question[idx])


def default_ego4d_narration():
    data_path = "data/ego4d/v2/nlq_official_v2_omnivore_video_fp16_official_128_bert.pkl"
    data = load_pickle(data_path)
    train_data = data["train_set"]
    val_data = data["val_set"]
    train_data.extend(val_data)

    # process answers
    narration_file_path = "data/ego4d/v2/annotations/narration.json"
    narration_data = load_json_file(narration_file_path)

    nlq_file_path1 = "data/ego4d/v2/annotations/nlq_train.json"
    nlq_data = load_json_file(nlq_file_path1)

    nlq_file_path2 = "data/ego4d/v2/annotations/nlq_val.json"
    nlq_data2 = load_json_file(nlq_file_path2)
    nlq_data["videos"].extend(nlq_data2["videos"])

    video_clip_dict = {}
    clip_id_list = []

    for video in nlq_data["videos"]:
        video_uid = video['video_uid']
        for clip in video['clips']:
            clip_uid = clip['clip_uid']
            clip_s_time = clip['video_start_sec']
            clip_e_time = clip['video_end_sec']
            video_clip_dict[clip_uid] = [video_uid, clip_s_time, clip_e_time]
    
    for i in tqdm.tqdm(range(len(train_data))):
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
        if "narration_pass_1" in video_clip:
            narrations += video_clip["narration_pass_1"]["narrations"]
        if "narration_pass_2" in video_clip:
            narrations += video_clip["narration_pass_2"]["narrations"]
        for narration in narrations:
            timestamp = narration['timestamp_sec']-clip_s_time
            timestamp_frame = narration['timestamp_frame']
            # assert timestamp>=0
            if timestamp>=0 and timestamp<=480:
                img_path = get_img_frames(clip_id, timestamp_frame)
                if img_path is not None:
                    questions.append("Give a caption.")


def default_planning_no_word():
    answers = []
    questions = []
    mask = []
    annotation = json.load(open("data/embodied/planning_val_v2.json", "r"))
    data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
    for idx, info in enumerate(data_info):
        scene_token = info['scene_token']
        timestamp = info['cams']['CAM_FRONT']['timestamp']
        image_path = info['cams']["CAM_FRONT"]['data_path']

        if scene_token not in annotation:
            continue
        value = annotation[scene_token]
        # scene_description = value['scene_description']
        scene_key_frame = value['key_frame']
        frame_id = str(timestamp)
        if frame_id in scene_key_frame:

            value1 = scene_key_frame[frame_id]

            if value1 is None:
                continue

            if "Planning" in value1:
                Planning_q = value1['Planning']['q']
                Planning_a = value1['Planning']['a']
                assert len(Planning_q) == len(Planning_a)
            else:
                Planning_q = []
                Planning_a = []

                
            Question = Planning_q
            Answer = Planning_a

            assert len(Question) == len(Answer)

            for idx in range(len(Question)):
                # reduce the decimals of questions
                x = Answer[idx].split(', ')
                x[0] = x[0].split('A: ')[-1]
                vocab = []
                combine = ""
                x = [t for t in x if t != ""]
                if len(x) != 12:
                    mask.append(0)
                    continue
                for j, num in enumerate(x):
                    if num == "":
                        continue
                    index = round(float(num), 1)
                    if index == 0.0:
                        index = 0.0
                    if j % 2 != 0:
                        combine += f"{index}]"
                        vocab.append(combine)
                        combine = ""
                    else:
                        combine += f"[{index}, "

                strings = ', '.join(vocab)
                strings = "Trajectory: " + strings

                questions.append(Question[idx])
                answers.append([strings])
                mask.append(1)

    return mask, questions, answers




if __name__ == "__main__":

    # For Drivelm
    mask, questions, answers = default_planning_no_word()
    import pdb; pdb.set_trace()
    with open("mask.json", 'w') as file:
        json.dump(mask, file)


    # # For Ego4d
    # get_ego4d()

    # # Moment Recap
    # nums = 0
    # for question in questions:
    #     # if "seconds before in the history" in question: # 6976
    #     #     continue
    #     if "What happened between" in question: # 6977
    #         continue
    #     if "What will happen in the next" in question: # 6895
    #         continue
    #     nums += 1
    # print("Moment Recap: ", nums)
    
    # # Event Query
    # nums = 0
    # for question in questions:
    #     if "seconds before in the history" in question: # 6976
    #         continue
    #     # if "What happened between" in question: # 6977
    #     #     continue
    #     if "What will happen in the next" in question: # 6895
    #         continue
    #     nums += 1
    # print("Event Query: ", nums)

    # # Future Activation
    # nums = 0
    # for question in questions:
    #     if "seconds before in the history" in question: # 6976
    #         continue
    #     if "What happened between" in question: # 6977
    #         continue
    #     # if "What will happen in the next" in question: # 6895
    #     #     continue
    #     nums += 1
    # print("Future Activation: ", nums)