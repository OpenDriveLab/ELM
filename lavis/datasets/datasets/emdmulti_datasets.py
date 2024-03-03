"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
The number of 0 questions:  17424   
The number of 1 questions:  5284                                                                                           
The number of 2 questions:  3285                                                                                               
The number of 3 questions:  1859                                                                                           
The number of 4 questions:  131                                                                                                          
The number of 5 questions:  26  
"""

import os
import json
import torch
import numpy as np
import pickle
import random

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from nuscenes.nuscenes import NuScenes
from collections import OrderedDict
import re
from lavis.datasets.data_utils import load_video_features, pad_video_seq, pad_seq, pad_char_seq, load_pickle
import tqdm


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


class EMDMULTIDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.answers = []
        self.questions = []
        self.images = []
        self.tmp_imglist = []
        self.traffic_element_dict = {}

        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.first_img = None

        # set temporal length
        self.temporal_length = 3

        # voc
        self.num_to_vocab = {}
        self.num_threshold = 30000

        with open('data/vocab.txt', 'r') as file:
            for line_number, line_content in enumerate(file, 1):
                line_content = line_content.strip()
                if line_number>=(self.num_threshold-1000):
                    self.num_to_vocab[line_number] = line_content


        self.default_boxqa(ann_paths)
        print("The number of total questions: ", len(self.questions))

        # self.default_future_boxqa_no_word(ann_paths)
        # print("The number of total questions: ", len(self.questions))

        # self.default_history_boxqa_no_word(ann_paths)
        # print("The number of total questions: ", len(self.questions))


    def default_planning_no_word(self):
        self.annotation = json.load(open("data/embodied/planning_train_v2.json", "r"))
        self.data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
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

                # temporal data only for image path
                tmp_image = []
                for tmp in range(0, self.temporal_length):
                    if (idx-tmp) >= 0 or scene_token != self.data_info[idx-tmp]['scene_token']:
                        continue
                    tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                    tmp_image.append(tmp_path)

                # if the image path is not equal self.temporal length, then use the duplicate image path
                tmp_image = tmp_image[::-1]
                if len(tmp_image) != self.temporal_length:
                    if len(tmp_image) != 0:
                        tmp_image = tmp_image + image_path * (self.temporal_length - len(tmp_image)) 
                    else:
                        tmp_image = [image_path] * self.temporal_length
                assert len(tmp_image) == self.temporal_length

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
                    if len(x) != 12:
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
                    self.questions.append(Question[idx])
                    self.answers.append([strings])
                    self.images.append(image_path)
                    self.tmp_imglist.append(tmp_image)


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

    def default_nusc(self, ann_paths):
        self.annotation = json.load(open(ann_paths[0], "r"))
        self.data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
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

                # temporal data only for image path
                tmp_image = []
                for tmp in range(0, self.temporal_length):
                    if scene_token != self.data_info[idx-tmp]['scene_token']:
                        continue
                    tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                    tmp_image.append(tmp_path)

                # if the image path is not equal self.temporal length, then use the duplicate image path
                tmp_image = tmp_image[::-1]
                if len(tmp_image) != self.temporal_length:
                    if len(tmp_image) != 0:
                        tmp_image = tmp_image[:1] * (self.temporal_length - len(tmp_image)) + tmp_image
                    else:
                        tmp_image = [image_path] * self.temporal_length
                assert len(tmp_image) == self.temporal_length

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
                self.tmp_imglist.append(tmp_image)


    def default_future_boxqa_no_word(self, ann_paths):
        self.annotation = json.load(open("data/embodied/Future_BOXQA_train.json", "r"))
        self.data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
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

                # temporal data only for image path
                tmp_image = []
                for tmp in range(1, self.temporal_length+1):
                    if scene_token != self.data_info[idx-tmp]['scene_token']:
                        continue
                    tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                    tmp_image.append(tmp_path)

                # if the image path is not equal self.temporal length, then use the duplicate image path
                tmp_image = tmp_image[::-1]
                if len(tmp_image) != self.temporal_length:
                    if len(tmp_image) != 0:
                        tmp_image = tmp_image[:1] * (self.temporal_length - len(tmp_image)) + tmp_image
                    else:
                        tmp_image = [image_path] * self.temporal_length
                assert len(tmp_image) == self.temporal_length

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

                self.questions.append(Question[idx])
                self.answers.append([strings])
                self.images.append(image_path)
                self.tmp_imglist.append(tmp_image)


    def default_history_boxqa_no_word(self, ann_paths):
        self.annotation = json.load(open("data/embodied/Hist_BOXQA_train.json", "r"))
        self.data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
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

                # temporal data only for image path
                tmp_image = []
                for tmp in range(1, self.temporal_length+1):
                    if scene_token != self.data_info[idx-tmp]['scene_token']:
                        continue
                    tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                    tmp_image.append(tmp_path)

                # if the image path is not equal self.temporal length, then use the duplicate image path
                tmp_image = tmp_image[::-1]
                if len(tmp_image) != self.temporal_length:
                    if len(tmp_image) != 0:
                        tmp_image = tmp_image[:1] * (self.temporal_length - len(tmp_image)) + tmp_image
                    else:
                        tmp_image = [image_path] * self.temporal_length
                assert len(tmp_image) == self.temporal_length

                value1 = scene_key_frame[frame_id]

                if value1 is None:
                    continue

                if "History BOX QA" in value1:
                    History_BOX_q = value1['History BOX QA']['q']
                    History_BOX_a = value1['History BOX QA']['a']
                    assert len(History_BOX_q) == len(History_BOX_a)
                else:
                    History_BOX_q = []
                    History_BOX_a = []

                    
                Question = History_BOX_q
                Answer = History_BOX_a

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

                self.questions.append(Question[idx])
                self.answers.append([strings])
                self.images.append(image_path)
                self.tmp_imglist.append(tmp_image)

    def default_boxqa_no_word(self, ann_paths):
        self.annotation = json.load(open("data/embodied/BOXQA_train_v3.json", "r"))
        self.data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
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

                # temporal data only for image path
                tmp_image = []
                for tmp in range(1, self.temporal_length+1):
                    if scene_token != self.data_info[idx-tmp]['scene_token']:
                        continue
                    tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                    tmp_image.append(tmp_path)

                # if the image path is not equal self.temporal length, then use the duplicate image path
                tmp_image = tmp_image[::-1]
                if len(tmp_image) != self.temporal_length:
                    if len(tmp_image) != 0:
                        tmp_image = tmp_image[:1] * (self.temporal_length - len(tmp_image)) + tmp_image
                    else:
                        tmp_image = [image_path] * self.temporal_length
                assert len(tmp_image) == self.temporal_length

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

                x, y, z, label = Answer[idx].split(', ')
                x = x.split('A: ')[-1]
                if round(float(y)) > 70:
                    continue
                index1 = f"{round(float(x), 1)}"
                index2 = f"{round(float(y), 1)}"
                index3 = f"{round(float(z), 1)}"

                vocab = [index1, index2, index3]
                strings = 'Location: '+' '.join(vocab) + ' Label: ' + label

                self.questions.append(Question[idx])
                self.answers.append([strings])
                self.images.append(image_path)
                self.tmp_imglist.append(tmp_image)

    def default_boxqa(self, ann_paths):
        self.annotation = json.load(open("data/embodied/BOXQA_train_v3.json", "r"))
        self.data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
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

                # temporal data only for image path
                tmp_image = []
                for tmp in range(1, self.temporal_length+1):
                    if scene_token != self.data_info[idx-tmp]['scene_token']:
                        continue
                    tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                    tmp_image.append(tmp_path)

                # if the image path is not equal self.temporal length, then use the duplicate image path
                tmp_image = tmp_image[::-1]
                if len(tmp_image) != self.temporal_length:
                    if len(tmp_image) != 0:
                        tmp_image = tmp_image[:1] * (self.temporal_length - len(tmp_image)) + tmp_image
                    else:
                        tmp_image = [image_path] * self.temporal_length
                assert len(tmp_image) == self.temporal_length

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

            if self.first_img is None:
                self.first_img = image_path

            for idx in range(len(Question)):
                # reduce the decimals of questions
                coor1, coor2 = Question[idx].split('<c, CAM_FRONT, ')[-1].split('>')[0].split(', ')
                num1, num2 = round(float(coor1), 1), round(float(coor2), 1)
                Question[idx] = Question[idx].replace(coor1, str(num1))
                Question[idx] = Question[idx].replace(coor2, str(num2))
                Question[idx] += ", then describe the class of this object."

                x, y, z, label = Answer[idx].split(', ')
                x = x.split('A: ')[-1]
                index = round(float(x)) + self.num_threshold 
                vocab_1 = self.num_to_vocab[index]

                index = round(float(y)) + self.num_threshold + 100 # !!!important!!!!!
                vocab_2 = self.num_to_vocab[index]

                index = round(float(z)) + self.num_threshold + 300
                vocab_3 = self.num_to_vocab[index]

                vocab = [vocab_1, vocab_2, vocab_3]
                strings = 'Location: '+' '.join(vocab) + ' Label: ' + label

                self.questions.append(Question[idx])
                self.answers.append([strings])
                self.images.append(self.first_img)
                self.tmp_imglist.append([self.first_img]*self.temporal_length)
        
    def __getitem__(self, index):
        image_path = self.images[index]
        tmp_imglist = self.tmp_imglist[index]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.questions[index]

        # # re
        # pattern = r'(\d+\.\d+) seconds (ago|later)'
        # matches = re.findall(pattern, question)
        # time_values = [match[0] for match in matches][0]
        # question1 = question.split(time_values)[1]
        # question = question.split(time_values)[0]

        question = self.text_processor(question)

        # question = question + " " + time_values + question1

        answer = self.answers[index]

        tmp_image = []
        for tmp_img in tmp_imglist:
            tmp = Image.open(tmp_img).convert("RGB")
            tmp = self.vis_processor(tmp)
            tmp_image.append(tmp)
        tmp_image = torch.stack(tmp_image, dim=0)
        
        return {
            "question": question,
            "answer": answer,
            "image": image,
            "tmp_image": tmp_image,
        }

    def __len__(self):
        return len(self.questions)

    def collater(self, samples):
        # merge samples into a list for each key
        questions = [s["question"] for s in samples]
        answers = [s["answer"] for s in samples]
        images = [s["image"] for s in samples]
        tmp_images = [s["tmp_image"] for s in samples]

        images = torch.stack(images, dim=0)
        tmp_images = torch.stack(tmp_images, dim=0)
        # [][][] -> []
        answers = [item[0] for item in answers]

        return {
            "vfeats": tmp_images,
            "questions": questions,
            "answers": answers,
            # "tmp_images": tmp_images,
        }


class EMDMULTIDatasetEvalDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.answers = []
        self.questions = []
        self.images = []
        self.tmp_imglist = []
        self.traffic_element_dict = {}
        self.first_img = None

        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        # set temporal length
        self.temporal_length = 3

        # voc
        self.num_to_vocab = {}
        self.num_threshold = 30000

        with open('data/vocab.txt', 'r') as file:
            for line_number, line_content in enumerate(file, 1):
                line_content = line_content.strip()
                if line_number>=(self.num_threshold-1000):
                    self.num_to_vocab[line_number] = line_content

        # self.default_future_boxqa_no_word(ann_paths)
        # print("The number of total questions: ", len(self.questions))
        # self.default_history_boxqa_no_word(ann_paths)
        # print("The number of total questions: ", len(self.questions))
        self.default_boxqa(ann_paths)
        print("The number of total questions: ", len(self.questions))


    def default_planning_waymo(self):
        self.annotation = json.load(open("data/embodied/planning_waymo.json", "r"))
        for idx, info in enumerate(self.annotation):
            image_path = info['image_path']
            question = info['questions']
            answer = info['answers'][0]

            x = answer.split(', ')
            x[0] = x[0].split('A: ')[-1]
            vocab = []
            combine = ""
            if len(x) != 12:
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
            self.questions.append(question)
            self.answers.append([strings + f"__{idx}"])
            self.images.append(image_path)
            self.tmp_imglist.append([image_path]*self.temporal_length)


    def default_planning_youtube(self):
        # for youtube folder data, find all images ,then every images with a answer
        root = "data/YouTube/V1/mini_images/J_Utah/yhCAWZlv_Sc_J_Utah-Brooklyn 4K - Night Drive"
        for img in os.listdir(root):
            self.images.append(os.path.join(root, img))
            speed = random.randint(1, 10)
            self.questions.append(f"Q: The ego car is moving forward at a speed of {speed}.78. Calculate 6 trajectory points into the future.")
            self.answers.append(["The car is driving on the road."])
            self.tmp_imglist.append([os.path.join(root, img)] * self.temporal_length)


    def default_planning_no_word(self):
        self.annotation = json.load(open("data/embodied/planning_val_v2.json", "r"))
        self.data_info = pickle.load(open("data/infos/nuscenes_infos_temporal_val.pkl", "rb"))["infos"]
        for idx, info in enumerate(self.data_info):
            scene_token = info['scene_token']
            timestamp = info['cams']['CAM_FRONT']['timestamp']
            image_path = os.path.join("data/nuscenes", info['cams']["CAM_FRONT"]['data_path'])

            if scene_token not in self.annotation:
                continue
            value = self.annotation[scene_token]
            # scene_description = value['scene_description']
            scene_key_frame = value['key_frame']
            frame_id = str(timestamp)
            if frame_id in scene_key_frame:

                # temporal data only for image path
                tmp_image = []
                for tmp in range(0, self.temporal_length):
                    if (idx-tmp) >= 0 or scene_token != self.data_info[idx-tmp]['scene_token']:
                        continue
                    tmp_path = os.path.join("data/nuscenes", self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path'])
                    tmp_image.append(tmp_path)

                # if the image path is not equal self.temporal length, then use the duplicate image path
                tmp_image = tmp_image[::-1]
                if len(tmp_image) != self.temporal_length:
                    if len(tmp_image) != 0:
                        tmp_image = tmp_image + image_path * (self.temporal_length - len(tmp_image)) 
                    else:
                        tmp_image = [image_path] * self.temporal_length
                assert len(tmp_image) == self.temporal_length

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

                for i in range(len(Question)):
                    # reduce the decimals of questions
                    x = Answer[i].split(', ')
                    x[0] = x[0].split('A: ')[-1]
                    vocab = []
                    combine = ""
                    if len(x) != 12:
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

                    self.questions.append(Question[i])
                    self.answers.append([strings + f"__{idx}"])
                    self.images.append(image_path)
                    self.tmp_imglist.append(tmp_image)



    def default_ego4d(self):
        data_path = "data/ego4d/v2/nlq_official_v2_omnivore_video_fp16_official_128_bert.pkl"
        data = load_pickle(data_path)
        self.val_data = data["val_set"]

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
        
        for i in tqdm.tqdm(range(len(self.val_data))):
            record = self.val_data[i]
            
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

    def default_nusc(self, ann_paths):
        self.annotation = json.load(open(ann_paths[0], "r"))
        self.data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
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

                # temporal data only for image path
                tmp_image = []
                for tmp in range(0, self.temporal_length):
                    if scene_token != self.data_info[idx-tmp]['scene_token']:
                        continue
                    tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                    tmp_image.append(tmp_path)

                # if the image path is not equal self.temporal length, then use the duplicate image path
                tmp_image = tmp_image[::-1]
                if len(tmp_image) != self.temporal_length:
                    if len(tmp_image) != 0:
                        tmp_image = tmp_image[:1] * (self.temporal_length - len(tmp_image)) + tmp_image
                    else:
                        tmp_image = [image_path] * self.temporal_length
                assert len(tmp_image) == self.temporal_length

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
                    
                Question = Prediction_q
                Answer = Prediction_a

            assert len(Question) == len(Answer)

            for idx in range(len(Question)):
                self.questions.append(Question[idx])
                self.answers.append([Answer[idx]])
                self.images.append(image_path)
                self.tmp_imglist.append(tmp_image)

    def default_future_boxqa_no_word(self, ann_paths):
        self.annotation = json.load(open("data/embodied/Future_BOXQA_val.json", "r"))
        self.data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_val.pkl", "rb"))["infos"]
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

                # temporal data only for image path
                tmp_image = []
                for tmp in range(0, self.temporal_length):
                    if (idx+tmp) >= len(self.data_info) or scene_token != self.data_info[idx-tmp]['scene_token']:
                        continue
                    tmp_path = self.data_info[idx+tmp]['cams']['CAM_FRONT']['data_path']
                    tmp_image.append(tmp_path)

                # if the image path is not equal self.temporal length, then use the duplicate image path
                tmp_image = tmp_image[::-1]
                if len(tmp_image) != self.temporal_length:
                    if len(tmp_image) != 0:
                        tmp_image = tmp_image[:1] * (self.temporal_length - len(tmp_image)) + tmp_image
                    else:
                        tmp_image = [image_path] * self.temporal_length
                assert len(tmp_image) == self.temporal_length

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

                self.questions.append(Question[idx])
                self.answers.append([strings])
                self.images.append(image_path)
                self.tmp_imglist.append(tmp_image)


    def default_history_boxqa_no_word(self, ann_paths):
        self.annotation = json.load(open("data/embodied/Hist_BOXQA_val.json", "r"))
        self.data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
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

                # temporal data only for image path
                tmp_image = []
                for tmp in range(0, self.temporal_length):
                    if (idx-tmp) < 0 or scene_token != self.data_info[idx-tmp]['scene_token']:
                        continue
                    tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                    tmp_image.append(tmp_path)

                # if the image path is not equal self.temporal length, then use the duplicate image path
                tmp_image = tmp_image[::-1]
                if len(tmp_image) != self.temporal_length:
                    if len(tmp_image) != 0:
                        tmp_image = tmp_image[:1] * (self.temporal_length - len(tmp_image)) + tmp_image
                    else:
                        tmp_image = [image_path] * self.temporal_length
                assert len(tmp_image) == self.temporal_length

                value1 = scene_key_frame[frame_id]

                if value1 is None:
                    continue

                if "History BOX QA" in value1:
                    History_BOX_q = value1['History BOX QA']['q']
                    History_BOX_a = value1['History BOX QA']['a']
                    assert len(History_BOX_q) == len(History_BOX_a)
                else:
                    History_BOX_q = []
                    History_BOX_a = []

                    
                Question = History_BOX_q
                Answer = History_BOX_a

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

                self.questions.append(Question[idx])
                self.answers.append([strings])
                self.images.append(image_path)
                self.tmp_imglist.append(tmp_image)

    def default_boxqa_no_word(self, ann_paths):
        self.annotation = json.load(open("data/embodied/BOXQA_val_v3.json", "r"))
        self.data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
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

                # temporal data only for image path
                tmp_image = []
                for tmp in range(1, self.temporal_length+1):
                    if scene_token != self.data_info[idx-tmp]['scene_token']:
                        continue
                    tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                    tmp_image.append(tmp_path)

                # if the image path is not equal self.temporal length, then use the duplicate image path
                tmp_image = tmp_image[::-1]
                if len(tmp_image) != self.temporal_length:
                    if len(tmp_image) != 0:
                        tmp_image = tmp_image[:1] * (self.temporal_length - len(tmp_image)) + tmp_image
                    else:
                        tmp_image = [image_path] * self.temporal_length
                assert len(tmp_image) == self.temporal_length

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

                x, y, z, label = Answer[idx].split(', ')
                x = x.split('A: ')[-1]
                if round(float(y)) > 70:
                    continue
                index1 = f"{round(float(x), 1)}"
                index2 = f"{round(float(y), 1)}"
                index3 = f"{round(float(z), 1)}"

                vocab = [index1, index2, index3]
                strings = 'Location: '+' '.join(vocab) + ' Label: ' + label

                self.questions.append(Question[idx])
                self.answers.append([strings])
                self.images.append(image_path)
                self.tmp_imglist.append(tmp_image)

    def default_boxqa(self, ann_paths):
        self.annotation = json.load(open("data/embodied/BOXQA_val_v3.json", "r"))
        self.data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
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

                # temporal data only for image path
                tmp_image = []
                for tmp in range(1, self.temporal_length+1):
                    if scene_token != self.data_info[idx-tmp]['scene_token']:
                        continue
                    tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                    tmp_image.append(tmp_path)

                # if the image path is not equal self.temporal length, then use the duplicate image path
                tmp_image = tmp_image[::-1]
                if len(tmp_image) != self.temporal_length:
                    if len(tmp_image) != 0:
                        tmp_image = tmp_image[:1] * (self.temporal_length - len(tmp_image)) + tmp_image
                    else:
                        tmp_image = [image_path] * self.temporal_length
                assert len(tmp_image) == self.temporal_length

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

            if self.first_img is None:
                self.first_img = image_path
                
            for idx in range(len(Question)):
                # reduce the decimals of questions
                coor1, coor2 = Question[idx].split('<c, CAM_FRONT, ')[-1].split('>')[0].split(', ')
                num1, num2 = round(float(coor1), 1), round(float(coor2), 1)
                Question[idx] = Question[idx].replace(coor1, str(num1))
                Question[idx] = Question[idx].replace(coor2, str(num2))
                Question[idx] += ", then describe the class of this object."

                x, y, z, label = Answer[idx].split(', ')
                x = x.split('A: ')[-1]
                index = round(float(x)) + self.num_threshold 
                vocab_1 = self.num_to_vocab[index]

                index = round(float(y)) + self.num_threshold + 100 # !!!important!!!!!
                vocab_2 = self.num_to_vocab[index]

                index = round(float(z)) + self.num_threshold + 300
                vocab_3 = self.num_to_vocab[index]

                vocab = [vocab_1, vocab_2, vocab_3]
                strings = 'Location: '+' '.join(vocab) + ' Label: ' + label

                self.questions.append(Question[idx])
                self.answers.append([strings])
                self.images.append(self.first_img)
                self.tmp_imglist.append([self.first_img]*self.temporal_length)
                
        
    def __getitem__(self, index):
        image_path = self.images[index]
        tmp_imglist = self.tmp_imglist[index]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.questions[index]
        
        # # re
        # pattern = r'(\d+\.\d+) seconds (ago|later)'
        # matches = re.findall(pattern, question)
        # time_values = [match[0] for match in matches][0]
        # question1 = question.split(time_values)[1]
        # question = question.split(time_values)[0]

        question = self.text_processor(question)

        # question = question + " " + time_values + question1

        answer = self.answers[index]

        tmp_image = []
        for tmp_img in tmp_imglist:
            tmp = Image.open(tmp_img).convert("RGB")
            tmp = self.vis_processor(tmp)
            tmp_image.append(tmp)
        tmp_image = torch.stack(tmp_image, dim=0)
        
        return {
            "question": question,
            "answer": answer,
            "image": image,
            "tmp_image": tmp_image,
        }

    def __len__(self):
        return len(self.questions) 

    def collater(self, samples):
        # merge samples into a list for each key
        questions = [s["question"] for s in samples]
        answers = [s["answer"] for s in samples]
        images = [s["image"] for s in samples]
        tmp_images = [s["tmp_image"] for s in samples]

        images = torch.stack(images, dim=0)
        tmp_images = torch.stack(tmp_images, dim=0)
        # [][][] -> []
        answers = [item[0] for item in answers]

        return {
            "vfeats": tmp_images,
            "questions": questions,
            "answers": answers,
            # "tmp_images":  tmp_images,
        }

