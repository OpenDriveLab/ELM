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


class COMBINEDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.answers = []
        self.questions = []
        self.images = []

        self.answers_loc = []
        self.questions_loc = []
        self.images_loc = []

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

        # # finetune
        # self.default_ego4d()
        # print("The number of ego4d: ", len(self.questions))
        # self.default_drivelm(ann_paths)
        # print("The number of drivelm: ", len(self.questions))

        # location
        self.default_nusc_pretrain_noword()
        print("The number of nusc_pretrain: ", len(self.questions_loc))

        # caption
        # self.default_nuscaption()
        # print("The number of nuscaption: ", len(self.questions_loc))
        # self.default_waymo()
        # print("The number of waymo: ", len(self.questions))
        # self.default_youtube()
        # print("The number of youtube: ", len(self.questions))
        
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


    def default_nusc_pretrain_noword(self):
        self.annotation = json.load(open("data/embodied/pretrain_train_refine.json", "r"))
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
                
                if "Pretrain Fore Loc" in value1:
                    Pretrain_fore_q = value1['Pretrain Fore Loc']['q']
                    Pretrain_fore_a = value1['Pretrain Fore Loc']['a']
                    assert len(Pretrain_fore_q) == len(Pretrain_fore_a)
                else:
                    Pretrain_fore_q = []
                    Pretrain_fore_a = []
                
                if "Pretrain Back Loc" in value1:
                    Pretrain_back_q = value1['Pretrain Back Loc']['q']
                    Pretrain_back_a = value1['Pretrain Back Loc']['a']
                    assert len(Pretrain_back_q) == len(Pretrain_back_a)
                else:
                    Pretrain_back_q = []
                    Pretrain_back_a = []
                    
                Question = Pretrain_fore_q + Pretrain_back_q
                Answer = Pretrain_fore_a + Pretrain_back_a

            assert len(Question) == len(Answer)

            for idx in range(len(Question)):
                # reduce the decimals of questions
                coor1, coor2 = Question[idx].split('<c, CAM_FRONT, ')[-1].split('>')[0].split(', ')
                num1, num2 = round(float(coor1), 1), round(float(coor2), 1)
                Question[idx] = Question[idx].replace(coor1, str(num1))
                Question[idx] = Question[idx].replace(coor2, str(num2))

                x, y, z = Answer[idx].split(', ')
                x = x.split('A: ')[-1]
                if round(float(y)) > 70:
                    continue
                index1 = f"{round(float(x), 1)}"
                index2 = f"{round(float(y), 1)}"
                index3 = f"{round(float(z), 1)}"

                vocab = [index1, index2, index3]
                strings = 'Location: '+' '.join(vocab)
                
                self.questions_loc.append(Question[idx])
                self.answers_loc.append([strings])
                self.images_loc.append(image_path)


    def default_nusc_pretrain(self):
        self.annotation = json.load(open("data/embodied/pretrain_train_refine.json", "r"))
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
                
                if "Pretrain Fore Loc" in value1:
                    Pretrain_fore_q = value1['Pretrain Fore Loc']['q']
                    Pretrain_fore_a = value1['Pretrain Fore Loc']['a']
                    assert len(Pretrain_fore_q) == len(Pretrain_fore_a)
                else:
                    Pretrain_fore_q = []
                    Pretrain_fore_a = []
                
                if "Pretrain Back Loc" in value1:
                    Pretrain_back_q = value1['Pretrain Back Loc']['q']
                    Pretrain_back_a = value1['Pretrain Back Loc']['a']
                    assert len(Pretrain_back_q) == len(Pretrain_back_a)
                else:
                    Pretrain_back_q = []
                    Pretrain_back_a = []
                    

                Question = Pretrain_fore_q + Pretrain_back_q
                Answer = Pretrain_fore_a + Pretrain_back_a

            assert len(Question) == len(Answer)

            for idx in range(len(Question)):
                # reduce the decimals of questions
                coor1, coor2 = Question[idx].split('<c, CAM_FRONT, ')[-1].split('>')[0].split(', ')
                num1, num2 = round(float(coor1), 1), round(float(coor2), 1)
                Question[idx] = Question[idx].replace(coor1, str(num1))
                Question[idx] = Question[idx].replace(coor2, str(num2))

                x, y, z = Answer[idx].split(', ')
                x = x.split('A: ')[-1]
                if round(float(y)) > 70:
                    continue
                index = round(float(x)) + self.num_threshold 
                vocab_1 = self.num_to_vocab[index]

                index = round(float(y)) + self.num_threshold + 100 # !!!important!!!!!
                vocab_2 = self.num_to_vocab[index]

                index = round(float(z)) + self.num_threshold + 300
                vocab_3 = self.num_to_vocab[index]

                vocab = [vocab_1, vocab_2, vocab_3]
                strings = 'Location: '+' '.join(vocab)
                
                self.questions_loc.append(Question[idx])
                self.answers_loc.append([strings])
                self.images_loc.append(image_path)

    def default_nuscaption(self):
        data_path = "data/pretrain_data/nuscenes_caption.json"
        data = json.load(open(data_path, 'r'))

        for idx in range(len(data["questions"])):
            self.questions.append(data["questions"][idx])
            self.answers.append([data["answers"][idx]])
            self.images.append(data["image_paths"][idx])

    def default_waymo(self):
        data_path = "data/pretrain_data/waymo_data.json"
        data = json.load(open(data_path, 'r'))

        for idx in range(len(data["questions"])):
            self.questions.append(data["questions"][idx])
            self.answers.append([data["answers"][idx]])
            self.images.append(data["image_paths"][idx])

    def default_youtube(self):
        data_path1 = "data/pretrain_data/youtube_caption.json"
        data_path2 = "data/pretrain_data/youtube.json"

        data1 = json.load(open(data_path1, 'r'))
        data2 = json.load(open(data_path2, 'r'))

        for idx in range(len(data1["questions"])):
            self.questions.append(data1["questions"][idx])
            self.answers.append([data1["answers"][idx]])
            self.images.append(data1["image_paths"][idx])

        for idx in range(len(data2["questions"])):
            self.questions.append(data2["questions"][idx])
            self.answers.append([data2["answers"][idx]])
            self.images.append(data2["image_paths"][idx])

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
        # if random.randint(0, 1):
        index = random.randint(0, len(self.questions_loc)-1)
        image_path = self.images_loc[index]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.questions_loc[index]
        question = self.text_processor(question)
        answer = self.answers_loc[index]
        
        # else:
        #     index = random.randint(0, len(self.questions)-1)
        #     image_path = self.images[index]
        #     image = Image.open(image_path).convert("RGB")
        #     image = self.vis_processor(image)
        #     question = self.questions[index]
        #     question = self.text_processor(question)
        #     answer = self.answers[index]

        return {
            "question": question,
            "answer": answer,
            "image": image
        }

    def __len__(self):
        return (len(self.questions)+len(self.questions_loc)) // 10

    def collater(self, samples):
        # merge samples into a list for each key
        questions = [s["question"] for s in samples]
        answers = [s["answer"] for s in samples]
        images = [s["image"] for s in samples]

        images = torch.stack(images, dim=0)
        # [][][] -> []
        answers = [item[0] for item in answers]

        return {
            "vfeats": images,
            "questions": questions,
            "answers": answers,
        }


class COMBINEDatasetEvalDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.answers = []
        self.questions = []
        self.images = []

        self.answers_loc = []
        self.questions_loc = []
        self.images_loc = []

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

        # # finetune
        # self.default_ego4d()
        # print("The number of ego4d: ", len(self.questions))
        # self.default_drivelm(ann_paths)
        # print("The number of drivelm: ", len(self.questions))

        # location
        self.default_nusc_pretrain_noword()
        print("The number of nusc_pretrain: ", len(self.questions_loc))

        # # youtube 
        # self.default_youtube()
        # print("The number of youbute: ", len(self.questions))


        # self.questions = self.questions[::10]
        # self.answers = self.answers[::10]
        # self.images = self.images[::10]

        self.questions.extend(self.questions_loc[::100])
        self.answers.extend(self.answers_loc[::100])
        self.images.extend(self.images_loc[::100])


    def default_youtube(self):
        data_path = "data/pretrain_data/youtube.json"
        data = json.load(open(data_path, 'r'))

        for idx in range(5000):
            idx = random.randint(0, 800000)
            self.questions.append(data["questions"][idx])
            self.answers.append([data["answers"][idx]])
            self.images.append(data["image_paths"][idx])

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


    def default_nusc_pretrain_noword(self):
        self.annotation = json.load(open("data/embodied/pretrain_val_refine.json", "r"))
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
                
                if "Pretrain Fore Loc" in value1:
                    Pretrain_fore_q = value1['Pretrain Fore Loc']['q']
                    Pretrain_fore_a = value1['Pretrain Fore Loc']['a']
                    assert len(Pretrain_fore_q) == len(Pretrain_fore_a)
                else:
                    Pretrain_fore_q = []
                    Pretrain_fore_a = []
                
                if "Pretrain Back Loc" in value1:
                    Pretrain_back_q = value1['Pretrain Back Loc']['q']
                    Pretrain_back_a = value1['Pretrain Back Loc']['a']
                    assert len(Pretrain_back_q) == len(Pretrain_back_a)
                else:
                    Pretrain_back_q = []
                    Pretrain_back_a = []
                    
                Question = Pretrain_fore_q + Pretrain_back_q
                Answer = Pretrain_fore_a + Pretrain_back_a

            assert len(Question) == len(Answer)

            for idx in range(len(Question)):
                # reduce the decimals of questions
                coor1, coor2 = Question[idx].split('<c, CAM_FRONT, ')[-1].split('>')[0].split(', ')
                num1, num2 = round(float(coor1), 1), round(float(coor2), 1)
                Question[idx] = Question[idx].replace(coor1, str(num1))
                Question[idx] = Question[idx].replace(coor2, str(num2))

                x, y, z = Answer[idx].split(', ')
                x = x.split('A: ')[-1]
                if round(float(y)) > 70:
                    continue
                index1 = f"{round(float(x), 1)}"
                index2 = f"{round(float(y), 1)}"
                index3 = f"{round(float(z), 1)}"

                vocab = [index1, index2, index3]
                strings = 'Location: '+' '.join(vocab)
                
                self.questions_loc.append(Question[idx])
                self.answers_loc.append([strings])
                self.images_loc.append(image_path)


    def default_nusc_pretrain(self):
        self.annotation = json.load(open("data/embodied/pretrain_val_refine.json", "r"))
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
                
                if "Pretrain Fore Loc" in value1:
                    Pretrain_fore_q = value1['Pretrain Fore Loc']['q']
                    Pretrain_fore_a = value1['Pretrain Fore Loc']['a']
                    assert len(Pretrain_fore_q) == len(Pretrain_fore_a)
                else:
                    Pretrain_fore_q = []
                    Pretrain_fore_a = []
                
                if "Pretrain Back Loc" in value1:
                    Pretrain_back_q = value1['Pretrain Back Loc']['q']
                    Pretrain_back_a = value1['Pretrain Back Loc']['a']
                    assert len(Pretrain_back_q) == len(Pretrain_back_a)
                else:
                    Pretrain_back_q = []
                    Pretrain_back_a = []
                    

                Question = Pretrain_fore_q + Pretrain_back_q
                Answer = Pretrain_fore_a + Pretrain_back_a

            
            assert len(Question) == len(Answer)

            for idx in range(len(Question)):
                # reduce the decimals of questions
                coor1, coor2 = Question[idx].split('<c, CAM_FRONT, ')[-1].split('>')[0].split(', ')
                num1, num2 = round(float(coor1), 1), round(float(coor2), 1)
                Question[idx] = Question[idx].replace(coor1, str(num1))
                Question[idx] = Question[idx].replace(coor2, str(num2))

                x, y, z = Answer[idx].split(', ')
                x = x.split('A: ')[-1]
                if round(float(y)) > 70:
                    continue
                index = round(float(x)) + self.num_threshold 
                vocab_1 = self.num_to_vocab[index]

                index = round(float(y)) + self.num_threshold + 100 # !!!important!!!!!
                vocab_2 = self.num_to_vocab[index]

                index = round(float(z)) + self.num_threshold + 300
                vocab_3 = self.num_to_vocab[index]

                vocab = [vocab_1, vocab_2, vocab_3]
                strings = 'Location: '+' '.join(vocab)
                
                self.questions_loc.append(Question[idx])
                self.answers_loc.append([strings])
                self.images_loc.append(image_path)


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
        # merge samples into a list for each key
        questions = [s["question"] for s in samples]
        answers = [s["answer"] for s in samples]
        images = [s["image"] for s in samples]

        images = torch.stack(images, dim=0)
        # [][][] -> []
        answers = [item[0] for item in answers]

        return {
            "vfeats": images,
            "questions": questions,
            "answers": answers,
        }


loc_x_bin = [-17.18, -13.36, -11.08, -9.17, -7.17, -5.64, -4.07, -2.91, -1.3, 0.61, 2.27, 3.25, 4.11, 5.06, 6.14, 7.45, 9.02, 11.21, 15.17] # 20
loc_y_bin = [9.69, 12.54, 15.09, 17.31, 19.48, 21.55, 23.56, 25.62, 27.56, 29.47, 31.42, 33.54, 35.68, 38.12, 40.66, 43.4, 46.42, 50.04, 54.52] # 20
loc_z_bin =  [-1.31, -0.94, -0.52, 0.03]    # 5
dim_l_bin =  [0.46, 0.56, 0.64, 0.73, 0.92, 1.75, 1.83, 1.89, 1.95, 2.01, 2.08, 2.21, 2.56, 3.05] # 15
dim_h_bin = [0.54, 0.82, 4.36, 4.82] # 5
dim_w_bin = [0.96, 1.12, 1.44, 1.57, 1.66, 1.75, 1.82, 1.93, 2.22]  # 10
yaw_bin = [-4.36, -3.30, -3.13, -2.95, -1.6, -1.16, -0.05, 0.04, 1.04]  # 10
velo_x_bin = [-0.05, 0.0, 0.05] # 4
velo_y_bin = [-0.07, 0.0, 0.09] # 4

def special_encode_split_bin(questions, answers):
    location_dict = ['far', 'depth', 'coordinate', 'distance', 'located', 'position','location', 'close', 'offset', 'space available', 'deep', 'deviate', 'Where', 'away from the ego car', '-axis value', 'further']
    dim_dict = ['size', 'tall', 'wide', 'long', 'length', 'width', 'height', 'dimension','high', 'big', 'ratio', 'large', 'shape', 'volume', 'thick']
    rad_dict = ['yaw', 'angle', 'radians', 'orientation']
    velo_dict = ['velocity', 'm/s', 'fast', 'speed', 'per second', 'velocities', 'slow', 'moving']
    pass_dict = ['does not specify', 'does not provide', 'does not indicate']
    return_answers = []
    returen_questions = []

    def find_bin(value, bin_boundaries):
        bin_index = next((i+1 for i, bin_boundary in enumerate(bin_boundaries) if value < bin_boundary), len(bin_boundaries)+1)
        return bin_index

    for idx in range(len(questions)):
        question, answer = questions[idx], answers[idx]

        text_without_brackets = re.sub(r'<[^>]+>', '', answer)
        pattern = r'[-]?\d+\.\d+'
        signed_floats = re.findall(pattern, text_without_brackets)
        encode_type = []
        flag = None

        for item in pass_dict:
            if item in question or item in answer:
                # return None
                continue

        for item in location_dict:
            if item in question or item in answer:
                encode_type.append("loc")
        for item in dim_dict:
            if item in question or item in answer:
                encode_type.append("dim")
                # encode_type.append("loc")
        for item in rad_dict:
            if item in question or item in answer:
                encode_type.append("rad")
                # encode_type.append("loc")
        for item in velo_dict:
            if item in question or item in answer:
                encode_type.append("velo")
                # encode_type.append("loc")
        
        if len(set(encode_type)) > 1:
            flag = 'hybrid'
        elif len(set(encode_type)) == 1:
            flag = encode_type[0]
        # if flag == None:
        #     return answer
        if flag == 'hybrid':
            continue
            # flag = "loc"
        num_answer = ""
        # if len(signed_floats) == 3:
        if flag == "loc" and len(signed_floats) == 3:
            num = [float(signed_floats[0]),float(signed_floats[1]),float(signed_floats[2])]
            # i = find_bin(num[0], loc_x_bin)//2 # x: [-17.18, 15.17] -> 20
            i = int((num[0]+18)//3.6)
            # word = "<extra_id_%d>" % i
            # word = "<bin_%d>" % i
            word = "%d" % num[0]
            # num_answer += word
            num_answer += str(i)+' '
            answer = answer.replace(signed_floats[0], word)
            # i = find_bin(num[1], loc_y_bin)//2 # y: [9.69, 54.52] -> 20
            i = int((num[1]-5)//5)
            # word = "<extra_id_%d>" % (i+10)
            # word = "<bin_%d>" % (i+10)
            word = "%d" % num[1]
            # num_answer += word
            num_answer += str(i+10)+' '
            answer = answer.replace(signed_floats[1], word)
            # i = find_bin(num[2], loc_z_bin) # z: [-1.31, 0.03] -> 5
            i = int((num[2]+1.31)//0.3)
            # word = "<extra_id_%d>" % (i+20)
            # word = "<bin_%d>" % (i+20)
            word = "%d" % num[2]
            # num_answer += word
            num_answer += str(i+20)+' '
            answer = answer.replace(signed_floats[2], word)

            # return_answers.append(num_answer)
            # return_answers.append(str(signed_floats))
            return_answers.append(str([int(float(d)) for d in signed_floats]))

            
            if num[1]>20 and num[1]<40 and abs(num[0]) <2:
                question += "It is at a moderate distance in front of the ego car."
            if num[1]<20 and abs(num[0]) <2:
                question += "It is not far ahead in front of the ego car."
            if num[1]>40 and abs(num[0]) <2:
                question += "It is at a far distance in front of the ego car."
            if num[1]>20 and num[1]<40 and num[0] > 2:
                question += "It is at a moderate distance to the right front of the ego car."
            if num[1]>40 and num[0] > 2:
                question += "It is at a far distance to the right front of the ego car."
            if num[1]<20 and num[0] > 2:
                question += "It is almost on the right side of the ego car."
            if num[1]>20 and num[1]<40 and num[0] < -2:
                question += "It is at a moderate distance to the left front of the ego car."
            if num[1]>40 and num[0] < -2:
                question += "It is at a far distance to the left front of the ego car."
            if num[1]<20 and num[0] < -2:
                question += "It is almost on the left side of the ego car."
            if num[0]>10:
                question += "It's extremly right to the ego car."
            if num[0]<-10:
                question += "It's extremly left to the ego car."

            # returen_questions.append(question+answers[idx])
            returen_questions.append(question)
        if flag == "dim" and len(signed_floats) == 3:
            num = [float(signed_floats[0]),float(signed_floats[1]),float(signed_floats[2])]
            i = find_bin(num[0], dim_l_bin) # l: [0.46, 3.05] -> 15
            # word = "<extra_id_%d>" % (i+25)
            word = "<bin_%d>" % (i+25)
            num_answer += word
            answer = answer.replace(signed_floats[0], word)
            i = find_bin(num[1], dim_h_bin) # h: [0.54, 4.82] -> 5
            # word = "<extra_id_%d>" % (i+40)
            word = "<bin_%d>" % (i+40)
            num_answer += word
            answer = answer.replace(signed_floats[1], word)
            i = find_bin(num[2], dim_w_bin) # w: [0.96, 2.22] -> 10
            # word = "<extra_id_%d>" % (i+45)
            word = "<bin_%d>" % (i+45)
            num_answer += word
            answer = answer.replace(signed_floats[2], word)

            # print(signed_floats, str([int(float(d)) for d in signed_floats]))
            # input()
            # return_answers.append(str([int(float(d)+1) for d in signed_floats]))
            # returen_questions.append(question)

        if flag == "rad" and len(signed_floats) == 1:
            num = [float(signed_floats[0])]
            i = find_bin(num[0], yaw_bin) # yaw: [-4.36, 1.04] -> 10
            # word = "<extra_id_%d>" % (i+55)
            word = "<bin_%d>" % (i+55)
            num_answer += word
            answer = answer.replace(signed_floats[0], word)
        if flag == "velo" and len(signed_floats) == 2:
            num = [float(signed_floats[0]),float(signed_floats[1])]
            i = find_bin(num[0], velo_x_bin) # velox: [-0.05, 0.05] -> 4
            # word = "<extra_id_%d>" % (i+65)
            word = "<bin_%d>" % (i+65)
            num_answer += word
            answer = answer.replace(signed_floats[0], word)
            i = find_bin(num[1], velo_y_bin) # veloy: [-0.07, 0.09] -> 4
            # word = "<extra_id_%d>" % (i+69)
            word = "<bin_%d>" % (i+69)
            num_answer += word
            answer = answer.replace(signed_floats[1], word)

        if num_answer == "":
            num_answer = answer
        # return_answers.append(num_answer)
        # # return_answers.append(answer)
        # # returen_questions.append(question)
        # returen_questions.append(question+answers[idx])
        # return_answers.append(str([int(float(d)) for d in signed_floats]))
        # returen_questions.append(question+answers[idx])
    
    shuffle_ans = return_answers
    # random.shuffle(shuffle_ans)
    # returen_questions = [question+str(shuffle_ans) for question in returen_questions]

    # return returen_questions[:3], return_answers[:3]
    # print(returen_questions, return_answers)
    # input()
    return returen_questions, return_answers