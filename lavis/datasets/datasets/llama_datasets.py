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

def get_img_frames(clip_id):
    img_dir = "data/ego4d/output_data"

    def get_numeric_part(filename):
        return int(filename.split('_')[1].split('.')[0])
    try:
        img_path = os.path.join(img_dir, clip_id)
        img_list = os.listdir(img_path)[:128]
        img_list = sorted(img_list, key=get_numeric_part)
        img_list = [os.path.join(img_path, img) for img in img_list]
    
        assert len(img_list) == 128
        return img_list
    except:
        return []

class LLAMADataset(VQADataset, __DisplMixin):
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
    
        self.default_nusc(ann_paths)
        print("The number of total questions: ", len(self.questions))

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
                    GPT_q, GPT_a = special_encode_split_bin(GPT_q, GPT_a)
                    # print(GPT_q, GPT_a)
                    # input()
                    assert len(GPT_q) == len(GPT_a)
                else:
                    GPT_q = []
                    GPT_a = []

                Question = Perception_q + Prediction_q #+ GPT_q
                Answer = Perception_a + Prediction_a #+ GPT_a

            else:
                continue
            
            assert len(Question) == len(Answer)

            for idx in range(len(Question)):
                self.questions.append(Question[idx])
                self.answers.append([Answer[idx]])
                self.images.append(image_path)
                
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
            "images": images,
            "questions": questions,
            "answers": answers,
        }


class LLAMADatasetEvalDataset(VQADataset, __DisplMixin):
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
    
        self.default_nusc(ann_paths)

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
                    # GPT_q, GPT_a = special_encode(GPT_q, GPT_a)
                    GPT_q, GPT_a = special_encode_split_bin(GPT_q, GPT_a)
                    # print(GPT_q, GPT_a)
                    # input()
                    assert len(GPT_q) == len(GPT_a)
                else:
                    GPT_q = []
                    GPT_a = []

                Question = Perception_q + Prediction_q #+ GPT_q
                Answer = Perception_a + Prediction_a #+ GPT_a
            else:
                continue
            
            assert len(Question) == len(Answer)

            for idx in range(len(Question)):
                self.questions.append(Question[idx])
                self.answers.append([Answer[idx]])
                self.images.append(image_path)
                

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
        return len(self.questions) // 5

    def collater(self, samples):
        # merge samples into a list for each key
        questions = [s["question"] for s in samples]
        answers = [s["answer"] for s in samples]
        images = [s["image"] for s in samples]

        images = torch.stack(images, dim=0)
        # [][][] -> []
        answers = [item[0] for item in answers]

        return {
            "images": images,
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