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


class ELMDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.answers = []
        self.questions = []
        self.images = []

        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        # voc
        self.num_to_vocab = {}
        self.num_threshold = 30000

        with open('data/vocab.txt', 'r') as file:
            for line_number, line_content in enumerate(file, 1):
                line_content = line_content.strip()
                if line_number>=(self.num_threshold-1000):
                    self.num_to_vocab[line_number] = line_content

        self.default_drivelm()
        print("The number of data: ", len(self.questions))


    def default_drivelm(self, ann_paths):
        self.annotation = json.load(open(ann_paths[0], "r"))
        self.data_info = pickle.load(open("data/nuscenes/bevdetv2-nuscenes_infos_train.pkl", "rb"))["infos"]
        for idx, info in enumerate(self.data_info):
            scene_token = info['scene_token']
            timestamp = info['cams']['CAM_FRONT']['timestamp']
            image_path = info['cams']["CAM_FRONT"]['data_path']

            if scene_token not in self.annotation:
                continue
            value = self.annotation[scene_token]
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
                                    

                Question = Perception_q + Prediction_q
                Answer = Perception_a + Prediction_a

            
            assert len(Question) == len(Answer)

            for idx in range(len(Question)):                
                self.questions.append(Question[idx])
                self.answers.append([Answer[idx]])
                self.images.append(image_path)


    def default_ego4d(self):
        data_path = "data/ego4d/v2/nlq_official_v2_omnivore_video_fp16_official_128_bert.pkl"
        data = load_pickle(data_path)
        self.train_data = data["train_set"]

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
        
        for i in tqdm.tqdm(range(len(self.train_data))):
            record = self.train_data[i]
            
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
                        self.questions.append("Give a caption.")
                        self.answers.append([narration['narration_text'][3:]])
                        self.images.append(img_path)
        
    def __getitem__(self, index):
        index = random.randint(0, len(self.questions)-1)
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.questions[index]
        question = self.text_processor(question)
        answer = self.answers[index]

        return {
            "question": question,
            "answer": answer,
            "image": image
        }

    def __len__(self):
        return (len(self.questions)+len(self.questions)) // 10

    def collater(self, samples):
        # merge samples into a list for each key
        questions = [s["question"] for s in samples]
        answers = [s["answer"] for s in samples]
        images = [s["image"] for s in samples]

        images = torch.stack(images, dim=0)
        answers = [item[0] for item in answers]

        return {
            "vfeats": images,
            "questions": questions,
            "answers": answers,
        }


class ELMDatasetEvalDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.answers = []
        self.questions = []
        self.images = []

        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        # voc
        self.num_to_vocab = {}
        self.num_threshold = 30000

        with open('data/vocab.txt', 'r') as file:
            for line_number, line_content in enumerate(file, 1):
                line_content = line_content.strip()
                if line_number>=(self.num_threshold-1000):
                    self.num_to_vocab[line_number] = line_content


        self.default_drivelm()
        print("The number of nusc_pretrain: ", len(self.questions))

        self.questions.extend(self.questions)
        self.answers.extend(self.answers)
        self.images.extend(self.images)


    def default_drivelm(self, ann_paths):
        self.annotation = json.load(open(ann_paths[0], "r"))
        self.data_info = pickle.load(open("data/nuscenes/bevdetv2-nuscenes_infos_val.pkl", "rb"))["infos"]
        for idx, info in enumerate(self.data_info):
            scene_token = info['scene_token']
            timestamp = info['cams']['CAM_FRONT']['timestamp']
            image_path = info['cams']["CAM_FRONT"]['data_path']

            if scene_token not in self.annotation:
                continue
            value = self.annotation[scene_token]
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
                                    

                Question = Perception_q + Prediction_q
                Answer = Perception_a + Prediction_a

            
            assert len(Question) == len(Answer)

            for idx in range(len(Question)):                
                self.questions.append(Question[idx])
                self.answers.append([Answer[idx]])
                self.images.append(image_path)



    def default_ego4d(self):
        data_path = "data/ego4d/v2/nlq_official_v2_omnivore_video_fp16_official_128_bert.pkl"
        data = load_pickle(data_path)
        self.train_data = data["val_set"]

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
        
        for i in tqdm.tqdm(range(len(self.train_data))):
            record = self.train_data[i]
            
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
                        self.questions.append("Give a caption.")
                        self.answers.append([narration['narration_text'][3:]])
                        self.images.append(img_path)


    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.questions[index]
        question = self.text_processor(question)
        answer = self.answers[index]

        return {
            "question": question,
            "answer": answer,
            "image": image
        }

    def __len__(self):
        return len(self.questions)

    def collater(self, samples):
        questions = [s["question"] for s in samples]
        answers = [s["answer"] for s in samples]
        images = [s["image"] for s in samples]

        images = torch.stack(images, dim=0)
        answers = [item[0] for item in answers]

        return {
            "vfeats": images,
            "questions": questions,
            "answers": answers,
        }

