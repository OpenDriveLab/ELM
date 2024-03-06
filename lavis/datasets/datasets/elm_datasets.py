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
        self.first_img = None

        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.num_to_vocab = {}
        self.num_threshold = 30000

        with open('data/vocab.txt', 'r') as file:
            for line_number, line_content in enumerate(file, 1):
                line_content = line_content.strip()
                if line_number>=(self.num_threshold-1000):
                    self.num_to_vocab[line_number] = line_content

        self.default_drivelm()
        print("The number of data: ", len(self.questions))


    def configure_traffic(self):
        data_root = 'data/openlane_v2_nus'
        with open(data_root + '/data_dict_subset_B_train.pkl', 'rb') as f:
            data_infos = pickle.load(f)
        
        data_infos = list(data_infos.values())
        num=0
        
        for info in tqdm(data_infos):
            ann_info = info['annotation']
            timestamp = info['timestamp']
            scene_id = info['meta_data']['source_id']

            if scene_id not in self.traffic_element_dict.keys():
                self.traffic_element_dict[scene_id] = {}

            gt_lanes = [np.array(lane['points'], dtype=np.float32) for lane in ann_info['lane_centerline']]
            te_bboxes = np.array([np.array(sign['points'], dtype=np.float32).flatten() for sign in ann_info['traffic_element']])
            if len(te_bboxes) == 0:
                te_bboxes = np.zeros((0, 4), dtype=np.float32)

            distance = []
            for i in range(len(gt_lanes)):
                distance.append((sum(sum(np.array(gt_lanes[i])**2))**0.5)/len(gt_lanes[i]))
            te_list = []
            for i in range(len(ann_info['traffic_element'])):
                te_ann = ann_info['traffic_element'][i]
                attribute = te_ann['attribute']
                if attribute>=4:
                    num+=1
                    te_list.append(attribute)
            
            self.traffic_element_dict[scene_id][timestamp] = te_list
        print("The number of total traffic elements: ", num)


    def default_traffic(self):
        self.data_info = pickle.load(open("data/nuscenes/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]

        neg_num = 0
        pos_num = 0
        for idx, info in enumerate(self.data_info):
            scene_token = info['scene_token']
            timestamp = info['cams']['CAM_FRONT']['timestamp']
            image_path = info['cams']["CAM_FRONT"]['data_path']
            if scene_token not in self.traffic_element_dict:
                continue   
            value = self.traffic_element_dict[scene_token]
            if timestamp in value:
                # temporal data only for image path
                tmp_image = []
                history_te = [self.traffic_element_dict[scene_token][timestamp]]
                for tmp in range(1, self.temporal_length+1):
                    if scene_token != self.data_info[idx-tmp]['scene_token']:
                        continue
                    tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                    tmp_image.append(tmp_path)

                    timestamp = self.data_info[idx-tmp]['cams']['CAM_FRONT']['timestamp']
                    scene_id = self.data_info[idx-tmp]['scene_token']
                    if scene_id not in self.traffic_element_dict.keys():
                        continue
                    if timestamp not in self.traffic_element_dict[scene_id].keys():
                        continue
                    history_te.append(self.traffic_element_dict[scene_id][timestamp])

                # if the image path is not equal self.temporal length, then use the duplicate image path
                tmp_image = tmp_image[::-1]
                if len(tmp_image) != self.temporal_length:
                    if len(tmp_image) != 0:
                        tmp_image = tmp_image[:1] * (self.temporal_length - len(tmp_image)) + tmp_image
                    else:
                        tmp_image = [image_path] * self.temporal_length
                assert len(tmp_image) == self.temporal_length

                # get the traffic element 
                longest_list = max(history_te, key=len)
                element_counts = {}
                for element in longest_list:
                    if element in element_counts:
                        element_counts[element] += 1
                    else:
                        element_counts[element] = 1

                length = len(element_counts)
                Traffic_q = 'Has the ego vehicle seen any traffic sign before?'
                if length==0:
                    Traffic_a = 'No. There is no traffic sign in the scene.'
                if length==1:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]} before.'
                if length==2:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]} and {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]} before.'
                if length==3:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]} and {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]} before.'
                if length==4:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]} and {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]} before.'
                if length==5:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]} and {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]} before.'
                if length==6:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]} and {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]} before.'
                if length==7:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]} and {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]} before.'
                if length==8:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]}, {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]} and {element_counts[longest_list[7]]} {self.te_convert[longest_list[7]]} before.'
                if length==9:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]}, {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]}, {element_counts[longest_list[7]]} {self.te_convert[longest_list[7]]} and {element_counts[longest_list[8]]} {self.te_convert[longest_list[8]]} before.'
                if length==10:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]}, {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]}, {element_counts[longest_list[7]]} {self.te_convert[longest_list[7]]}, {element_counts[longest_list[8]]} {self.te_convert[longest_list[8]]} and {element_counts[longest_list[9]]} {self.te_convert[longest_list[9]]} before.'
                if length==11:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]}, {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]}, {element_counts[longest_list[7]]} {self.te_convert[longest_list[7]]}, {element_counts[longest_list[8]]} {self.te_convert[longest_list[8]]}, {element_counts[longest_list[9]]} {self.te_convert[longest_list[9]]} and {element_counts[longest_list[10]]} {self.te_convert[longest_list[10]]} before.'
                if length==12:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]}, {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]}, {element_counts[longest_list[7]]} {self.te_convert[longest_list[7]]}, {element_counts[longest_list[8]]} {self.te_convert[longest_list[8]]}, {element_counts[longest_list[9]]} {self.te_convert[longest_list[9]]}, {element_counts[longest_list[10]]} {self.te_convert[longest_list[10]]} and {element_counts[longest_list[11]]} {self.te_convert[longest_list[11]]} before.'
                if length==13:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]}, {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]}, {element_counts[longest_list[7]]} {self.te_convert[longest_list[7]]}, {element_counts[longest_list[8]]} {self.te_convert[longest_list[8]]}, {element_counts[longest_list[9]]} {self.te_convert[longest_list[9]]}, {element_counts[longest_list[10]]} {self.te_convert[longest_list[10]]}, {element_counts[longest_list[11]]} {self.te_convert[longest_list[11]]} and {element_counts[longest_list[12]]} {self.te_convert[longest_list[12]]} before.'
                if length==14:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]}, {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]}, {element_counts[longest_list[7]]} {self.te_convert[longest_list[7]]}, {element_counts[longest_list[8]]} {self.te_convert[longest_list[8]]}, {element_counts[longest_list[9]]} {self.te_convert[longest_list[9]]}, {element_counts[longest_list[10]]} {self.te_convert[longest_list[10]]}, {element_counts[longest_list[11]]} {self.te_convert[longest_list[11]]}, {element_counts[longest_list[12]]} {self.te_convert[longest_list[12]]} and {element_counts[longest_list[13]]} {self.te_convert[longest_list[13]]} before.'
                if length==15:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]}, {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]}, {element_counts[longest_list[7]]} {self.te_convert[longest_list[7]]}, {element_counts[longest_list[8]]} {self.te_convert[longest_list[8]]}, {element_counts[longest_list[9]]} {self.te_convert[longest_list[9]]}, {element_counts[longest_list[10]]} {self.te_convert[longest_list[10]]}, {element_counts[longest_list[11]]} {self.te_convert[longest_list[11]]}, {element_counts[longest_list[12]]} {self.te_convert[longest_list[12]]}, {element_counts[longest_list[13]]} {self.te_convert[longest_list[13]]} and {element_counts[longest_list[14]]} {self.te_convert[longest_list[14]]} before.'

            if (length == 0 and idx % 3 == 0) or (length != 0):
                if length == 0:
                    neg_num += 1
                else:
                    pos_num += 1
                self.images.append(image_path)
                self.tmp_imglist.append(tmp_image)
                self.questions.append(Traffic_q)
                self.answers.append([Traffic_a])
        
        print("The number of traffic questions: ", len(self.questions))
        print("The number of positive questions: ", pos_num)
        print("The number of negative questions: ", neg_num)


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

    def default_boxqa(self, ann_paths):
        self.annotation = json.load(open("data/embodied/BOXQA_train_v3.json", "r"))
        self.data_info = pickle.load(open("data/nuscenes/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
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
        question = self.text_processor(question)
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
        return (len(self.questions)+len(self.questions))

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
            "images": images,
            "questions": questions,
            "answers": answers,
            "vfeats": tmp_images,
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

        self.num_to_vocab = {}
        self.num_threshold = 30000

        with open('data/vocab.txt', 'r') as file:
            for line_number, line_content in enumerate(file, 1):
                line_content = line_content.strip()
                if line_number>=(self.num_threshold-1000):
                    self.num_to_vocab[line_number] = line_content


        self.default_drivelm()
        print("The number of data: ", len(self.questions))

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

    def default_boxqa(self, ann_paths):
        self.annotation = json.load(open("data/embodied/BOXQA_val_v3.json", "r"))
        self.data_info = pickle.load(open("data/nuscenes/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
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


    def configure_traffic(self):
        data_root = 'data/openlane_v2_nus'
        with open(data_root + '/data_dict_subset_B_val.pkl', 'rb') as f:
            data_infos = pickle.load(f)
        
        data_infos = list(data_infos.values())
        num=0
        
        for info in tqdm(data_infos):
            ann_info = info['annotation']
            timestamp = info['timestamp']
            scene_id = info['meta_data']['source_id']

            if scene_id not in self.traffic_element_dict.keys():
                self.traffic_element_dict[scene_id] = {}

            gt_lanes = [np.array(lane['points'], dtype=np.float32) for lane in ann_info['lane_centerline']]
            te_bboxes = np.array([np.array(sign['points'], dtype=np.float32).flatten() for sign in ann_info['traffic_element']])
            if len(te_bboxes) == 0:
                te_bboxes = np.zeros((0, 4), dtype=np.float32)

            distance = []
            for i in range(len(gt_lanes)):
                distance.append((sum(sum(np.array(gt_lanes[i])**2))**0.5)/len(gt_lanes[i]))
            te_list = []
            for i in range(len(ann_info['traffic_element'])):
                te_ann = ann_info['traffic_element'][i]
                attribute = te_ann['attribute']
                if attribute>=4:
                    num+=1
                    te_list.append(attribute)
            
            self.traffic_element_dict[scene_id][timestamp] = te_list
        print("The number of total traffic elements: ", num)


    def default_traffic(self):
        self.data_info = pickle.load(open("data/nuscenes/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]

        neg_num = 0
        pos_num = 0
        for idx, info in enumerate(self.data_info):
            scene_token = info['scene_token']
            timestamp = info['cams']['CAM_FRONT']['timestamp']
            image_path = info['cams']["CAM_FRONT"]['data_path']
            if scene_token not in self.traffic_element_dict:
                continue   
            value = self.traffic_element_dict[scene_token]
            if timestamp in value:
                # temporal data only for image path
                tmp_image = []
                history_te = [self.traffic_element_dict[scene_token][timestamp]]
                for tmp in range(1, self.temporal_length+1):
                    if scene_token != self.data_info[idx-tmp]['scene_token']:
                        continue
                    tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                    tmp_image.append(tmp_path)

                    timestamp = self.data_info[idx-tmp]['cams']['CAM_FRONT']['timestamp']
                    scene_id = self.data_info[idx-tmp]['scene_token']
                    if scene_id not in self.traffic_element_dict.keys():
                        continue
                    if timestamp not in self.traffic_element_dict[scene_id].keys():
                        continue
                    history_te.append(self.traffic_element_dict[scene_id][timestamp])

                # if the image path is not equal self.temporal length, then use the duplicate image path
                tmp_image = tmp_image[::-1]
                if len(tmp_image) != self.temporal_length:
                    if len(tmp_image) != 0:
                        tmp_image = tmp_image[:1] * (self.temporal_length - len(tmp_image)) + tmp_image
                    else:
                        tmp_image = [image_path] * self.temporal_length
                assert len(tmp_image) == self.temporal_length

                # get the traffic element 
                longest_list = max(history_te, key=len)
                element_counts = {}
                for element in longest_list:
                    if element in element_counts:
                        element_counts[element] += 1
                    else:
                        element_counts[element] = 1

                length = len(element_counts)
                Traffic_q = 'Has the ego vehicle seen any traffic sign before?'
                if length==0:
                    Traffic_a = 'No. There is no traffic sign in the scene.'
                if length==1:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]} before.'
                if length==2:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]} and {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]} before.'
                if length==3:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]} and {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]} before.'
                if length==4:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]} and {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]} before.'
                if length==5:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]} and {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]} before.'
                if length==6:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]} and {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]} before.'
                if length==7:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]} and {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]} before.'
                if length==8:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]}, {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]} and {element_counts[longest_list[7]]} {self.te_convert[longest_list[7]]} before.'
                if length==9:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]}, {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]}, {element_counts[longest_list[7]]} {self.te_convert[longest_list[7]]} and {element_counts[longest_list[8]]} {self.te_convert[longest_list[8]]} before.'
                if length==10:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]}, {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]}, {element_counts[longest_list[7]]} {self.te_convert[longest_list[7]]}, {element_counts[longest_list[8]]} {self.te_convert[longest_list[8]]} and {element_counts[longest_list[9]]} {self.te_convert[longest_list[9]]} before.'
                if length==11:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]}, {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]}, {element_counts[longest_list[7]]} {self.te_convert[longest_list[7]]}, {element_counts[longest_list[8]]} {self.te_convert[longest_list[8]]}, {element_counts[longest_list[9]]} {self.te_convert[longest_list[9]]} and {element_counts[longest_list[10]]} {self.te_convert[longest_list[10]]} before.'
                if length==12:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]}, {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]}, {element_counts[longest_list[7]]} {self.te_convert[longest_list[7]]}, {element_counts[longest_list[8]]} {self.te_convert[longest_list[8]]}, {element_counts[longest_list[9]]} {self.te_convert[longest_list[9]]}, {element_counts[longest_list[10]]} {self.te_convert[longest_list[10]]} and {element_counts[longest_list[11]]} {self.te_convert[longest_list[11]]} before.'
                if length==13:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]}, {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]}, {element_counts[longest_list[7]]} {self.te_convert[longest_list[7]]}, {element_counts[longest_list[8]]} {self.te_convert[longest_list[8]]}, {element_counts[longest_list[9]]} {self.te_convert[longest_list[9]]}, {element_counts[longest_list[10]]} {self.te_convert[longest_list[10]]}, {element_counts[longest_list[11]]} {self.te_convert[longest_list[11]]} and {element_counts[longest_list[12]]} {self.te_convert[longest_list[12]]} before.'
                if length==14:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]}, {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]}, {element_counts[longest_list[7]]} {self.te_convert[longest_list[7]]}, {element_counts[longest_list[8]]} {self.te_convert[longest_list[8]]}, {element_counts[longest_list[9]]} {self.te_convert[longest_list[9]]}, {element_counts[longest_list[10]]} {self.te_convert[longest_list[10]]}, {element_counts[longest_list[11]]} {self.te_convert[longest_list[11]]}, {element_counts[longest_list[12]]} {self.te_convert[longest_list[12]]} and {element_counts[longest_list[13]]} {self.te_convert[longest_list[13]]} before.'
                if length==15:
                    Traffic_a = f'Yes. The ego vehicle has seen {element_counts[longest_list[0]]} {self.te_convert[longest_list[0]]}, {element_counts[longest_list[1]]} {self.te_convert[longest_list[1]]}, {element_counts[longest_list[2]]} {self.te_convert[longest_list[2]]}, {element_counts[longest_list[3]]} {self.te_convert[longest_list[3]]}, {element_counts[longest_list[4]]} {self.te_convert[longest_list[4]]}, {element_counts[longest_list[5]]} {self.te_convert[longest_list[5]]}, {element_counts[longest_list[6]]} {self.te_convert[longest_list[6]]}, {element_counts[longest_list[7]]} {self.te_convert[longest_list[7]]}, {element_counts[longest_list[8]]} {self.te_convert[longest_list[8]]}, {element_counts[longest_list[9]]} {self.te_convert[longest_list[9]]}, {element_counts[longest_list[10]]} {self.te_convert[longest_list[10]]}, {element_counts[longest_list[11]]} {self.te_convert[longest_list[11]]}, {element_counts[longest_list[12]]} {self.te_convert[longest_list[12]]}, {element_counts[longest_list[13]]} {self.te_convert[longest_list[13]]} and {element_counts[longest_list[14]]} {self.te_convert[longest_list[14]]} before.'

            if (length == 0 and idx % 3 == 0) or (length != 0):
                if length == 0:
                    neg_num += 1
                else:
                    pos_num += 1
                self.images.append(image_path)
                self.tmp_imglist.append(tmp_image)
                self.questions.append(Traffic_q)
                self.answers.append([Traffic_a])
        
        print("The number of traffic questions: ", len(self.questions))
        print("The number of positive questions: ", pos_num)
        print("The number of negative questions: ", neg_num)


    def __getitem__(self, index):
        image_path = self.images[index]
        tmp_imglist = self.tmp_imglist[index]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.questions[index]
        question = self.text_processor(question)
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
            "images": images,
            "questions": questions,
            "answers": answers,
            "vfeats": tmp_images,
        }
