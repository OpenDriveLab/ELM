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

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from nuscenes.nuscenes import NuScenes
from collections import OrderedDict
from tqdm import tqdm



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

class TRAFFICDataset(VQADataset, __DisplMixin):
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

        # set temporal length
        self.temporal_length = 14
        self.te_convert = {4: 'go_straight', 5: 'turn_left', 6: 'turn_right', 7: 'no_left_turn', 8: 'no_right_turn', 9: 'u_turn', 10: 'no_u_turn', 11: 'slight_left', 12: 'slight_right'}

        # self.configure_traffic()
        # self.default_traffic()
        # print("The number of traffic questions: ", len(self.questions))
        self.default_nusc(ann_paths)
        print("The number of total questions: ", len(self.questions))
    
    def configure_traffic(self):
        data_root = '/cpfs01/shared/opendrivelab/openlane_v2_nus'
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
        self.data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]

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
                
                if "Future Track" in value1:
                    Future_q = value1['Future Track']['q']
                    Future_a = value1['Future Track']['a']
                else:
                    Future_q = []
                    Future_a = []
                
                if "History Track" in value1:
                    History_q = value1['History Track']['q']
                    History_a = value1['History Track']['a']
                    # History_q, History_a = special_encode_split_bin(History_q, History_a)
                else:
                    History_q = []
                    History_a = []

                Question = Perception_q + Prediction_q
                Answer = Perception_a + Prediction_a

            else:
                continue
            
            assert len(Question) == len(Answer)

            for idx in range(len(Question)):
                self.questions.append(Question[idx])
                self.answers.append([Answer[idx]])
                self.images.append(image_path)
                self.tmp_imglist.append(tmp_image)

        
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


class TRAFFICDatasetEvalDataset(VQADataset, __DisplMixin):
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
        self.temporal_length = 14
        self.te_convert = {4: 'go_straight', 5: 'turn_left', 6: 'turn_right', 7: 'no_left_turn', 8: 'no_right_turn', 9: 'u_turn', 10: 'no_u_turn', 11: 'slight_left', 12: 'slight_right'}

        # self.configure_traffic()
        # self.default_traffic()
        # print("The number of traffic questions: ", len(self.questions))
        self.default_nusc(ann_paths)
        print("The number of total questions: ", len(self.questions))
    
    def configure_traffic(self):
        data_root = '/cpfs01/shared/opendrivelab/openlane_v2_nus'
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
        self.data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]

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


    def default_nusc(self, ann_paths):
        self.annotation = json.load(open(ann_paths[0], "r"))
        self.data_info = pickle.load(open("/cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/bevdetv2-nuscenes_infos_trainval.pkl", "rb"))["infos"]
        keep = []
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

                if "Perception" in value1:
                    Perception_q = value1['Perception']['q']
                    Perception_a = value1['Perception']['a']
                    keep.extend([0] * len(Perception_a))
                else:
                    Perception_q = []
                    Perception_a = []

                if "Prediction and Planning" in value1:
                    Prediction_q = value1['Prediction and Planning']['q']
                    Prediction_a = value1['Prediction and Planning']['a']
                    keep.extend([1] * len(Prediction_a))
                else:
                    Prediction_q = []
                    Prediction_a = []
                
                if "Future Track" in value1:
                    Future_q = value1['Future Track']['q']
                    Future_a = value1['Future Track']['a']
                else:
                    Future_q = []
                    Future_a = []
                
                if "History Track" in value1:
                    History_q = value1['History Track']['q']
                    History_a = value1['History Track']['a']
                    # History_q, History_a = special_encode_split_bin(History_q, History_a)
                else:
                    History_q = []
                    History_a = []

                Question = Prediction_q # Perception_q #+ Prediction_q
                Answer = Prediction_a # Perception_a #+ Prediction_a

            else:
                continue
            
            assert len(Question) == len(Answer)

            if self.first_img is None:
                self.first_img = image_path

            for idx in range(len(Question)):
                self.questions.append(Question[idx])
                self.answers.append([Answer[idx]])
                self.images.append(self.first_img)
                self.tmp_imglist.append([self.first_img]*self.temporal_length)
        
        self.questions = self.questions[::10]
        self.answers = self.answers[::10]
        self.images = self.images[::10]
        self.tmp_imglist = self.tmp_imglist[::10]
        # json.dump(keep, open("keep.json", "w"))

        
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

import re
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