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

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answer": "; ".join(ann["answers"]),
                "pc_feat": sample["pc_feat"],
                "pc": sample["pc"],
            }
        )

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



class LRNARRATIONDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        data_path = "data/ego4d/v2/nlq_official_v2_omnivore_video_fp16_official_128_bert.pkl"
        data = load_pickle(data_path)

        self.train_data = data["train_set"]
        # self.video_features = load_video_features(
        #     os.path.join("data/ego4d/v2", "nlq_official_v2_omnivore_video_fp16", "official"), 128)
        # self.keys = list(self.video_features.keys())[0]

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        # process answers
        narration_file_path = "data/ego4d/v2/annotations/narration.json"
        narration_data = load_json_file(narration_file_path)

        nlq_file_path = "data/ego4d/v2/annotations/nlq_train.json"
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
        
        self.data_infos = []
        self.questions = []
        self.answers = []
        self.time = []
        for i in tqdm.tqdm(range(len(self.train_data))):
            record = self.train_data[i]
            # print(record)
            # input()
            
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
            # img_frames = get_img_frames(clip_id)
            narrations = []
            # # image narration is here:
            # if "narration_pass_1" in video_clip:
            #     narrations += video_clip["narration_pass_1"]["narrations"]
            # if "narration_pass_2" in video_clip:
            #     narrations += video_clip["narration_pass_2"]["narrations"]
            # for narration in narrations:
            #     timestamp = narration['timestamp_sec']-clip_s_time
            #     timestamp_frame = narration['timestamp_frame']
            #     # assert timestamp>=0
            #     if timestamp>=0 and timestamp<=480:
            #         img_path = get_img_frames(clip_id, timestamp_frame)
            #         self.questions.append("Give a caption.")
            #         self.answers.append([narration['narration_text'][3:]])
            #         self.data_infos.append(img_path)
            if "narration_pass_1" in video_clip:
                narrations = video_clip["narration_pass_1"]["narrations"]
            else:
                if "narration_pass_2" in video_clip:
                    narrations = video_clip["narration_pass_2"]["narrations"]
                else:
                    continue
            narrations = sorted(narrations, key=lambda x: x['timestamp_sec'])
            # # memory narration is here:
            # mem_length = 5
            # for i in range(0, len(narrations)-mem_length, mem_length):
            #     narration_tmp = narrations[i:i+mem_length]
            #     answer = []
            #     img_path = []
            #     time = []
            #     if narration_tmp[-1]['timestamp_sec']<narration_tmp[0]['timestamp_sec']:
            #         continue
            #     for narration in narration_tmp:
            #         timestamp = narration['timestamp_sec']-clip_s_time
            #         timestamp_frame = narration['timestamp_frame']
            #         if timestamp>=0 and timestamp<=480:
            #             img_file = get_img_frames(clip_id, timestamp_frame)
            #             if img_file is not None:
            #                 img_path.append(img_file)
            #                 answer.append(narration['narration_text'][3:])
            #                 time.append(timestamp)
            #     if len(img_path) == mem_length:
            #         question = "Give a step-by-step description."
            #         answer = ' '.join(answer)
            #         self.questions.append(question)
            #         self.answers.append([answer])
            #         self.data_infos.append(img_path)
            #         self.time.append([t-time[-1] for t in time])
            
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
                    # question = f"What happened after '{narr[0]}'?"
                    answer = narr[1]
                    time = [0 for _ in time]
                    # time = [1 if (i>=start_index and i<start_index+3) else 0 for i in range(len(time))]
                    self.questions.append(question)
                    self.answers.append([answer])
                    self.data_infos.append(img_path)
                    self.time.append([t-time[-1] for t in time])
                    # self.time.append(time)

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
                    # question = f"Now '{answer[-1]}', what happened {his_time} seconds before in the history?"
                    answer = narr
                    # time = [1 if (i>=start_index and i<start_index+3) else 0 for i in range(len(time))]
                    self.questions.append(question)
                    self.answers.append([answer])
                    self.data_infos.append(img_path)
                    self.time.append([t-time[-1] for t in time])
                    # self.time.append(time)

            # # future query is here:
            # mem_length = 6
            # for i in range(0, len(narrations)-mem_length, mem_length):
            #     narration_tmp = narrations[i:i+mem_length]
            #     answer = []
            #     img_path = []
            #     time = []
            #     if narration_tmp[-1]['timestamp_sec']<narration_tmp[0]['timestamp_sec']:
            #         continue
            #     for narration in narration_tmp:
            #         timestamp = narration['timestamp_sec']-clip_s_time
            #         timestamp_frame = narration['timestamp_frame']
            #         if timestamp>=0 and timestamp<=480:
            #             img_file = get_img_frames(clip_id, timestamp_frame)
            #             if img_file is not None:
            #                 img_path.append(img_file)
            #                 answer.append(narration['narration_text'][3:])
            #                 time.append(timestamp)
            #     if len(img_path) == mem_length:
            #         question = "What will happen in the future?"
            #         answer = answer[-1]
            #         time = time[:-1]
            #         self.questions.append(question)
            #         self.answers.append([answer])
            #         self.data_infos.append(img_path[:mem_length-1])
            #         self.time.append([t-time[-1] for t in time])
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
                    # question = f"Now '{answer[-2]}', what will happen in the next {fu_time} seconds in the future?"
                    answer = answer[-1]
                    # time = [1 if (i>=len(time)-4) else 0 for i in range(len(time))]
                    time = time[:-1]
                    self.questions.append(question)
                    self.answers.append([answer])
                    self.data_infos.append(img_path[:mem_length-1])
                    self.time.append([t-time[-1] for t in time])
                    # self.time.append(time)
    

    def __getitem__(self, index):
        record = self.data_infos[index]
        # video_feature = self.video_features[record["vid"]]#[self.keys] #[record["vid"]]
        # image_path = self.data_infos[index]
        # image = Image.open(image_path).convert("RGB")
        # image = self.vis_processor(image)
        # video_feature = image
        adj_images = []
        for adj_path in self.data_infos[index]:
            adj_image = Image.open(adj_path).convert("RGB")
            adj_image = self.vis_processor(adj_image)
            adj_images.append(adj_image)
        adj_images = torch.stack(adj_images, dim=0)
        video_feature = adj_images
        # s_ind, e_ind = int(record["s_ind"]), int(record["e_ind"])
        # word_ids = record["w_ids"]
        # char_ids = record.get("c_ids", None)
        question = self.questions[index]
        answer = self.answers[index]
        time = torch.tensor(self.time[index])
        # print(time,question,answer)


        return {
            "record": record,
            "video_feature": video_feature,
            # "word_ids": word_ids,
            # "char_ids": char_ids,
            # "s_ind": s_ind,
            # "e_ind": e_ind,
            "question": question,
            "answer": answer,
            "time": time,
        }

    def __len__(self):
        return len(self.data_infos)

    def collater(self, samples):

        image_list, question_list, answer_list, weight_list, time_list = [], [], [], [], []

        num_answers = []

        for sample in samples:
            image_list.append(sample["video_feature"])
            question_list.append(sample["question"])
            answers = sample["answer"]

            answer_list.extend(answers)
            num_answers.append(len(answers))
            time_list.append(sample["time"])

        return {
            "records": None,
            "vfeats": torch.stack(image_list, dim=0),
            "vfeat_lens": None,
            "word_ids": None,
            "char_ids": None,
            "s_labels": None,
            "e_labels": None,
            "h_labels": None,
            "questions": question_list,
            "answers": answer_list,
            "timesteps": torch.stack(time_list, dim=0),
        }



class LRNARRATIONDatasetEvalDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        data_path = "data/ego4d/v2/nlq_official_v2_omnivore_video_fp16_official_128_bert.pkl"
        data = load_pickle(data_path)

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.val_data = data["val_set"]
        # self.video_features = load_video_features(
        #     os.path.join("data/ego4d/v2", "nlq_official_v2_omnivore_video_fp16", "official"), 128)
        # self.keys = list(self.video_features.keys())[0]

        # process answers
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
        
        self.questions = []
        self.answers = []
        self.data_infos = []
        self.time = []
        for i in tqdm.tqdm(range(len(self.val_data))):
            record = self.val_data[i]
            # print(record)
            # input()
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
            # # image narration is here:
            # if "narration_pass_1" in video_clip:
            #     narrations += video_clip["narration_pass_1"]["narrations"]
            # if "narration_pass_2" in video_clip:
            #     narrations += video_clip["narration_pass_2"]["narrations"]
            # for narration in narrations:
            #     timestamp = narration['timestamp_sec']-clip_s_time
            #     timestamp_frame = narration['timestamp_frame']
            #     # assert timestamp>=0
            #     if timestamp>=0 and timestamp<=480:
            #         img_path = get_img_frames(clip_id, timestamp_frame)
            #         self.questions.append("Give a caption.")
            #         self.answers.append([narration['narration_text'][3:]])
            #         self.data_infos.append(img_path)
            if "narration_pass_1" in video_clip:
                narrations = video_clip["narration_pass_1"]["narrations"]
            else:
                if "narration_pass_2" in video_clip:
                    narrations = video_clip["narration_pass_2"]["narrations"]
                else:
                    continue
            narrations = sorted(narrations, key=lambda x: x['timestamp_sec'])
            # # memory narration is here:
            # mem_length = 5
            # for i in range(0, len(narrations)-mem_length, mem_length):
            #     narration_tmp = narrations[i:i+mem_length]
            #     answer = []
            #     img_path = []
            #     time = []
            #     if narration_tmp[-1]['timestamp_sec']<narration_tmp[0]['timestamp_sec']:
            #         continue
            #     for narration in narration_tmp:
            #         timestamp = narration['timestamp_sec']-clip_s_time
            #         timestamp_frame = narration['timestamp_frame']
            #         if timestamp>=0 and timestamp<=480:
            #             img_file = get_img_frames(clip_id, timestamp_frame)
            #             if img_file is not None:
            #                 img_path.append(img_file)
            #                 answer.append(narration['narration_text'][3:])
            #                 time.append(timestamp)
            #     if len(img_path) == mem_length:
            #         question = "Give a step-by-step description."
            #         answer = ' '.join(answer)
            #         self.questions.append(question)
            #         self.answers.append([answer])
            #         self.data_infos.append(img_path)
            #         self.time.append([t-time[-1] for t in time])
            
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
                    # question = f"What happened after '{narr[0]}'?"
                    answer = narr[1]
                    time = [0 for _ in time]
                    # time = [1 if (i>=start_index and i<start_index+3) else 0 for i in range(len(time))]
                    self.questions.append(question)
                    self.answers.append([answer])
                    self.data_infos.append(img_path)
                    self.time.append([t-time[-1] for t in time])
                    # self.time.append(time)

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
                    # question = f"Now '{answer[-1]}', what happened {his_time} seconds before in the history?"
                    answer = narr
                    # time = [1 if (i>=start_index and i<start_index+3) else 0 for i in range(len(time))]
                    self.questions.append(question)
                    self.answers.append([answer])
                    self.data_infos.append(img_path)
                    self.time.append([t-time[-1] for t in time])
                    # self.time.append(time)

            # # future query is here:
            # mem_length = 6
            # for i in range(0, len(narrations)-mem_length, mem_length):
            #     narration_tmp = narrations[i:i+mem_length]
            #     answer = []
            #     img_path = []
            #     time = []
            #     if narration_tmp[-1]['timestamp_sec']<narration_tmp[0]['timestamp_sec']:
            #         continue
            #     for narration in narration_tmp:
            #         timestamp = narration['timestamp_sec']-clip_s_time
            #         timestamp_frame = narration['timestamp_frame']
            #         if timestamp>=0 and timestamp<=480:
            #             img_file = get_img_frames(clip_id, timestamp_frame)
            #             if img_file is not None:
            #                 img_path.append(img_file)
            #                 answer.append(narration['narration_text'][3:])
            #                 time.append(timestamp)
            #     if len(img_path) == mem_length:
            #         question = "What will happen in the future?"
            #         answer = answer[-1]
            #         time = time[:-1]
            #         self.questions.append(question)
            #         self.answers.append([answer])
            #         self.data_infos.append(img_path[:mem_length-1])
            #         self.time.append([t-time[-1] for t in time])
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
                    # question = f"Now '{answer[-2]}', what will happen in the next {fu_time} seconds in the future?"
                    answer = answer[-1]
                    # time = [1 if (i>=len(time)-4) else 0 for i in range(len(time))]
                    time = time[:-1]
                    self.questions.append(question)
                    self.answers.append([answer])
                    self.data_infos.append(img_path[:mem_length-1])
                    self.time.append([t-time[-1] for t in time])
                    # self.time.append(time)
        
        # self.questions = self.questions[::5]
        # self.answers = self.answers[::5]
        # self.data_infos = self.data_infos[::5]
        # self.time = self.time[::5]
        

    def __getitem__(self, index):
        record = self.data_infos[index]
        # video_feature = self.video_features[record["vid"]]#[self.keys] #[record["vid"]]
        # image_path = self.data_infos[index]
        # image = Image.open(image_path).convert("RGB")
        # image = self.vis_processor(image)
        # video_feature = image
        adj_images = []
        for adj_path in self.data_infos[index]:
            adj_image = Image.open(adj_path).convert("RGB")
            adj_image = self.vis_processor(adj_image)
            adj_images.append(adj_image)
        adj_images = torch.stack(adj_images, dim=0)
        video_feature = adj_images
        # s_ind, e_ind = int(record["s_ind"]), int(record["e_ind"])
        # word_ids = record["w_ids"]
        # char_ids = record.get("c_ids", None)
        question = self.questions[index]
        answer = self.answers[index]
        time = torch.tensor(self.time[index])



        return {
            "record": record,
            "video_feature": video_feature,
            # "word_ids": word_ids,
            # "char_ids": char_ids,
            # "s_ind": s_ind,
            # "e_ind": e_ind,
            "question": question,
            "answer": answer,
            "time": time,
        }

    def __len__(self):
        return len(self.data_infos)

    def collater(self, samples):
        image_list, question_list, answer_list, weight_list, time_list = [], [], [], [], []

        num_answers = []

        for sample in samples:
            image_list.append(sample["video_feature"])
            question_list.append(sample["question"])
            answers = sample["answer"]

            answer_list.extend(answers)
            num_answers.append(len(answers))
            time_list.append(sample["time"])

        return {
            "records": None,
            "vfeats": torch.stack(image_list, dim=0),
            "vfeat_lens": None,
            "word_ids": None,
            "char_ids": None,
            "s_labels": None,
            "e_labels": None,
            "h_labels": None,
            "questions": question_list,
            "answers": answer_list,
            "timesteps": torch.stack(time_list, dim=0),
        }
