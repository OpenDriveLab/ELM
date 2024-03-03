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
from nuscenes.nuscenes import NuScenes
import re
from collections import OrderedDict

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


class ThreeDVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.temporal_length = 3
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.annotation = json.load(open(ann_paths[0], "r"))

        self.data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
        # get the nusc_info
        nusc_info = {}
        annotation_list = []
        for idx, info in enumerate(self.data_info):
            scene_token = info['scene_token']
            timestamp = info['cams']['CAM_FRONT']['timestamp']
            image_path = info['cams']["CAM_FRONT"]['data_path']
            # get the temporal length image path from past frames
            adj_image_path, adj_id = [], []
            for i in range(1, self.temporal_length):
                if scene_token == self.data_info[idx - i]['scene_token'] and (idx - i) >= 0:
                    adj_image_path.append(self.data_info[idx - i]['cams']["CAM_FRONT"]['data_path'])
                    adj_id.append(scene_token+'/'+str(self.data_info[idx - i]['cams']["CAM_FRONT"]['timestamp']))
                else:
                    adj_image_path.append(image_path)
                    adj_id.append(scene_token+'/'+str(timestamp))

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

                if "GPT" in value1:
                    GPT_q = value1['GPT']['q']
                    GPT_a = value1['GPT']['a']
                    GPT_q, GPT_a = special_encode(GPT_q, GPT_a)
                    assert len(GPT_q) == len(GPT_a)
                else:
                    GPT_q = []
                    GPT_a = []

                Question = Perception_q + Prediction_q + GPT_q
                Answer = Perception_a + Prediction_a + GPT_a
            else:
                Question = ["Default"]
                Answer = ["None"]
            
            assert len(Question) == len(Answer)

            for idx in range(len(Question)):

                frame_dict = {
                    'scene_id': scene_token,
                    'frame_id': frame_id,
                    'question': Question[idx],
                    'answers': [Answer[idx]],
                    'image_path': image_path,
                    'adj_image_path': adj_image_path,
                    'adj_id': adj_id,
                }

                annotation_list.append(frame_dict)
        self.annotation = annotation_list

    

    def __getitem__(self, index):
        ann = self.annotation[index]
        caption = self.text_processor(ann["question"])
        scene_id = ann["scene_id"]
        frame_id = ann["frame_id"]

        image_path = ann["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        adj_images = []
        for adj_path in ann["adj_image_path"]:
            adj_image = Image.open(adj_path).convert("RGB")
            adj_image = self.vis_processor(adj_image)
            adj_images.append(adj_image)
        adj_images = torch.stack(adj_images, dim=0)

        answer_weight = {}
        for answer in ann["answers"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answers"])
            else:
                answer_weight[answer] = 1 / len(ann["answers"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "3d_feat": None,
            "image": image,
            "text_input": caption,
            "answers": answers,
            "weights": weights,
            "scene_id": (scene_id+'/'+frame_id),
            "question_id": index,
            "adj_images": adj_images,
            "adj_id": ann["adj_id"],
        }

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        image_list, question_list, answer_list, weight_list, adj_image_list, scene_id_list, adj_id_list = [], [], [], [], [], [], []

        num_answers = []

        for sample in samples:
            image_list.append(sample["image"])
            adj_image_list.append(sample["adj_images"])
            question_list.append(sample["text_input"])

            weight_list.extend(sample["weights"])
            answers = sample["answers"]

            answer_list.extend(answers)
            num_answers.append(len(answers))
            scene_id_list.append(sample["scene_id"])
            adj_id_list.append(sample["adj_id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "answer": answer_list,
            "weight": torch.Tensor(weight_list),
            "n_answers": torch.LongTensor(num_answers),
            "adj_images": torch.stack(adj_image_list, dim=0),
            "scene_id": scene_id_list,
            "adj_id": adj_id_list,
        }


class ThreeDVQAEvalDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.temporal_length = 3
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.annotation = json.load(open(ann_paths[0], "r"))

        self.data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
        # get the nusc_info
        annotation_list = []
        for idx, info in enumerate(self.data_info):
            scene_token = info['scene_token']
            timestamp = info['cams']['CAM_FRONT']['timestamp']
            image_path = info['cams']["CAM_FRONT"]['data_path']
            # get the temporal length image path from past frames
            adj_image_path, adj_id = [], []
            for i in range(1, self.temporal_length):
                if scene_token == self.data_info[idx - i]['scene_token'] and (idx - i) >= 0:
                    adj_image_path.append(self.data_info[idx - i]['cams']["CAM_FRONT"]['data_path'])
                    adj_id.append(scene_token+'/'+str(self.data_info[idx - i]['cams']["CAM_FRONT"]['timestamp']))
                else:
                    adj_image_path.append(image_path)
                    adj_id.append(scene_token+'/'+str(timestamp))

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

                if "GPT" in value1:
                    GPT_q = value1['GPT']['q']
                    GPT_a = value1['GPT']['a']
                    GPT_q, GPT_a = special_encode(GPT_q, GPT_a)
                    assert len(GPT_q) == len(GPT_a)
                else:
                    GPT_q = []
                    GPT_a = []

                Question = Perception_q + Prediction_q + GPT_q
                Answer = Perception_a + Prediction_a + GPT_a
            else:
                Question = ["Default"]
                Answer = ["None"]
            
            assert len(Question) == len(Answer)

            for idx in range(len(Question)):

                frame_dict = {
                    'scene_id': scene_token,
                    'frame_id': frame_id,
                    'question': Question[idx],
                    'answers': [Answer[idx]],
                    'image_path': image_path,
                    'adj_image_path': adj_image_path,
                    'adj_id': adj_id,
                }

                annotation_list.append(frame_dict)
        self.annotation = annotation_list

    def __getitem__(self, index):
        ann = self.annotation[index]
        caption = self.text_processor(ann["question"])
        scene_id = ann["scene_id"]
        frame_id = ann["frame_id"]

        image_path = ann["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        adj_images = []
        for adj_path in ann["adj_image_path"]:
            adj_image = Image.open(adj_path).convert("RGB")
            adj_image = self.vis_processor(adj_image)
            adj_images.append(adj_image)
        adj_images = torch.stack(adj_images, dim=0)

        answer_weight = {}
        for answer in ann["answers"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answers"])
            else:
                answer_weight[answer] = 1 / len(ann["answers"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "3d_feat": None,
            "image": image,
            "text_input": caption,
            "answers": answers,
            "weights": weights,
            "scene_id": (scene_id+'/'+frame_id),
            "question_id": index,
            "adj_images": adj_images,
            "adj_id": ann["adj_id"],
        }


    def __getitem__(self, index):
        ann = self.annotation[index]
        caption = self.text_processor(ann["question"])
        scene_id = ann["scene_id"]
        frame_id = ann["frame_id"]

        image_path = ann["image_path"]

        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        adj_images = []
        for adj_path in ann["adj_image_path"]:
            adj_image = Image.open(adj_path).convert("RGB")
            adj_image = self.vis_processor(adj_image)
            adj_images.append(adj_image)
        adj_images = torch.stack(adj_images, dim=0)

        return {
            "3d_feat": None,
            "image": image,
            "text_input": caption,
            "answers": ann["answers"],
            "scene_id": (scene_id+'/'+frame_id),
            "question_id": index,
            "adj_images": adj_images,
            "adj_id": ann["adj_id"],
        }

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        image_list, question_list, answer_list, question_id_list, adj_image_list, scene_id_list, adj_id_list = [], [], [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            adj_image_list.append(sample["adj_images"])
            question_list.append(sample["text_input"])
            question_id_list.append(sample["question_id"])
            answers = sample["answers"]
            answer_list.extend(answers)
            scene_id_list.append(sample["scene_id"])
            adj_id_list.append(sample["adj_id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "answer": answer_list,
            "question_id": torch.LongTensor(question_id_list),
            "adj_images": torch.stack(adj_image_list, dim=0),
            "scene_id": scene_id_list,
            "adj_id": adj_id_list,
        }

def special_encode(questions, answers):
    location_dict = ['far', 'depth', 'coordinate', 'distance', 'located', 'position','location', 'close', 'offset', 'space available', 'deep', 'deviate', 'Where', 'away from the ego car', '-axis value', 'further']
    dim_dict = ['size', 'tall', 'wide', 'long', 'length', 'width', 'height', 'dimension','high', 'big', 'ratio', 'large', 'shape', 'volume', 'thick']
    rad_dict = ['yaw', 'angle', 'radians', 'orientation']
    velo_dict = ['velocity', 'm/s', 'fast', 'speed', 'per second', 'velocities', 'slow', 'moving']
    pass_dict = ['does not specify', 'does not provide', 'does not indicate']
    return_answers = []
    returen_questions = []

    for i in range(len(questions)):
        question, answer = questions[i], answers[i]

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
                # encode_type.append("dim")
                encode_type.append("loc")
        for item in rad_dict:
            if item in question or item in answer:
                encode_type.append("rad")
        for item in velo_dict:
            if item in question or item in answer:
                # encode_type.append("velo")
                encode_type.append("loc")
        
        if len(set(encode_type)) > 1:
            flag = 'hybrid'
        elif len(set(encode_type)) == 1:
            flag = encode_type[0]
        # if flag == None:
        #     return answer
        if flag == 'hybrid':
            continue
            # flag = "loc"
        
        for signed_float in signed_floats:
            if flag == "loc":
                num = float(signed_float)
                i = (num+40) // (0.2) # [-40, 62]
                if i>=510:
                    continue
                word = "<loc%d>" % i
            if flag == "dim":
                num = float(signed_float)
                i = (num) // (0.1)
                assert(i<64)
                word = "<dim%d>" % i
            if flag == "rad":
                num = float(signed_float)
                i = (num+4.7124) // (0.1) # [-4,7124, 1.5708]
                # assert(i<64)
                if i>=64:
                    continue
                word = "<rad%d>" % i
            if flag == "velo":
                num = float(signed_float)
                i = (num+3.2) // (0.1)
                word = "<velo%d>" % i
                # num = float(signed_float)
                # i = (num+40) // (0.2) # [-40, 62]
                # assert(i<510)
                # word = "<loc%d>" % i
            if flag == None:
                # print(question, answer)
                pass
            else:
                answer = answer.replace(signed_float, word)
        return_answers.append(answer)
        returen_questions.append(question)
    
    return returen_questions, return_answers
