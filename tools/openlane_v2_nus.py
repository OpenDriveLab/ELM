import sys
import os
import json
import pickle
from glob import glob
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
# import mmcv

data_root = 'data/nuscenes/openlane_v2_nus'
with open(data_root + '/data_dict_subset_B_train.pkl', 'rb') as f:
    data_infos = pickle.load(f)
with open(data_root + '/data_dict_subset_B_val.pkl', 'rb') as f:
    data_infos.update(pickle.load(f))

TRAFFIC_ELEMENT_ATTRIBUTE = {
    'unknown':          0,
    'red':              1,
    'green':            2,
    'yellow':           3,
    'go_straight':      4,
    'turn_left':        5,
    'turn_right':       6,
    'no_left_turn':     7,
    'no_right_turn':    8,
    'u_turn':           9,
    'no_u_turn':        10,
    'slight_left':      11,
    'slight_right':     12,
}

GT_COLOR = (0, 255, 0)
PRED_COLOR = (0, 0, 255)

COLOR_DICT = {
    0:  (0, 0, 255),
    1:  (255, 0, 0),
    2:  (0, 255, 0),
    3:  (255, 255, 0),
    4:  (255, 0, 255),
    5:  (0, 128, 128),
    6:  (0, 128, 0),
    7:  (128, 0, 0),
    8:  (128, 0, 128),
    9:  (128, 128, 0),
    10: (0, 0, 128),
    11: (64, 64, 64),
    12: (192, 192, 192),
}


def _render_surround_img(images):
    all_image = []
    img_height = images[1].shape[0]

    for idx in [2, 0, 1, 5, 3, 4]:
        if idx == 4 or idx == 1:
            all_image.append(images[idx])
        else:
            all_image.append(images[idx])
            all_image.append(np.full((img_height, 10, 3), (255, 255, 255), dtype=np.uint8))

    surround_img_upper = None
    surround_img_upper = np.concatenate(all_image[:5], 1)

    surround_img_down = None
    surround_img_down = np.concatenate(all_image[5:], 1)

    surround_img = np.concatenate((surround_img_upper, np.full((10, surround_img_down.shape[1], 3), (255, 255, 255), dtype=np.uint8), surround_img_down), 0)
    surround_img = cv2.resize(surround_img, None, fx=0.5, fy=0.5)

    return surround_img

def show_results(image_list, lidar2imgs, gt_lane, pred_lane, gt_te=None, pred_te=None):
    res_image_list = []
    for idx, (raw_img, lidar2img) in enumerate(zip(image_list, lidar2imgs)):
        image = raw_img.copy()
        for lane in gt_lane:
            xyz1 = np.concatenate([lane, np.ones((lane.shape[0], 1))], axis=1)
            xyz1 = xyz1 @ lidar2img.T
            xyz1 = xyz1[xyz1[:, 2] > 1e-5]
            if xyz1.shape[0] == 0:
                continue
            points_2d = xyz1[:, :2] / xyz1[:, 2:3]
            points_2d = points_2d.astype(int)
            image = cv2.polylines(image, points_2d[None], False, GT_COLOR, 2)

        for lane in pred_lane:
            xyz1 = np.concatenate([lane, np.ones((lane.shape[0], 1))], axis=1)
            xyz1 = xyz1 @ lidar2img.T
            xyz1 = xyz1[xyz1[:, 2] > 1e-5]
            if xyz1.shape[0] == 0:
                continue
            points_2d = xyz1[:, :2] / xyz1[:, 2:3]
            points_2d = points_2d.astype(int)
            image = cv2.polylines(image, points_2d[None], False, PRED_COLOR, 2)

        if idx == 0:
            if gt_te is not None:
                for bbox, attr in gt_te:
                    b = bbox.astype(int)
                    color = COLOR_DICT[attr]
                    image = draw_corner_rectangle(image, (b[0], b[1]), (b[2], b[3]), color, 3, 1)
            if pred_te is not None:
                for bbox, attr in pred_te:
                    b = bbox.astype(int)
                    color = COLOR_DICT[attr]
                    image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, 3)

        res_image_list.append(image)

    return res_image_list

def draw_corner_rectangle(img: np.ndarray, pt1: tuple, pt2: tuple, color: tuple, corner_thickness: int = 3, edge_thickness: int = 2, centre_cross: bool = False, lineType: int = cv2.LINE_8):

    corner_length = min(abs(pt1[0] - pt2[0]), abs(pt1[1] - pt2[1])) // 4
    e_args = [color, edge_thickness, lineType]
    c_args = [color, corner_thickness, lineType]

    # edges
    img = cv2.line(img, (pt1[0] + corner_length, pt1[1]), (pt2[0] - corner_length, pt1[1]), *e_args)
    img = cv2.line(img, (pt2[0], pt1[1] + corner_length), (pt2[0], pt2[1] - corner_length), *e_args)
    img = cv2.line(img, (pt1[0], pt1[1] + corner_length), (pt1[0], pt2[1] - corner_length), *e_args)
    img = cv2.line(img, (pt1[0] + corner_length, pt2[1]), (pt2[0] - corner_length, pt2[1]), *e_args)
    # corners
    img = cv2.line(img, pt1, (pt1[0] + corner_length, pt1[1]), *c_args)
    img = cv2.line(img, pt1, (pt1[0], pt1[1] + corner_length), *c_args)
    img = cv2.line(img, (pt2[0], pt1[1]), (pt2[0] - corner_length, pt1[1]), *c_args)
    img = cv2.line(img, (pt2[0], pt1[1]), (pt2[0], pt1[1] + corner_length), *c_args)
    img = cv2.line(img, (pt1[0], pt2[1]), (pt1[0] + corner_length, pt2[1]), *c_args)
    img = cv2.line(img, (pt1[0], pt2[1]), (pt1[0], pt2[1] - corner_length), *c_args)
    img = cv2.line(img, pt2, (pt2[0] - corner_length, pt2[1]), *c_args)
    img = cv2.line(img, pt2, (pt2[0], pt2[1] - corner_length), *c_args)

    if centre_cross:
        cx, cy = int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2)
        img = cv2.line(img, (cx - corner_length, cy), (cx + corner_length, cy), *e_args)
        img = cv2.line(img, (cx, cy - corner_length), (cx, cy + corner_length), *e_args)

    return img

data_infos = list(data_infos.values())[:100]

# missing labels in the followed scenes.
BLACKLIST_SCENES = ('20ec831deb0f44e397497198cbe5a97c')
data_infos = [info for info in data_infos if info['meta_data']['source_id'] not in BLACKLIST_SCENES]

num=0
traffic_element_dict = {}
for info in tqdm(data_infos):

    
    ann_info = info['annotation']
    # the timestamp of the image is aligned to front view image.
    timestamp = info['timestamp']

    # the nuScenes scene token
    scene_id = info['meta_data']['source_id']

    if scene_id not in traffic_element_dict.keys():
        traffic_element_dict[scene_id] = {}

    gt_lanes = [np.array(lane['points'], dtype=np.float32) for lane in ann_info['lane_centerline']]
    lane_adj = np.array(ann_info['topology_lclc'], dtype=np.float32)

    # only use traffic light attribute
    te_bboxes = np.array([np.array(sign['points'], dtype=np.float32).flatten() for sign in ann_info['traffic_element']])
    te_labels = np.array([sign['attribute'] for sign in ann_info['traffic_element']], dtype=np.int64)
    if len(te_bboxes) == 0:
        te_bboxes = np.zeros((0, 4), dtype=np.float32)
        te_labels = np.zeros((0, ), dtype=np.int64)

    lane_lcte_adj = np.array(ann_info['topology_lcte'], dtype=np.float32)

    distance = []
    for i in range(len(gt_lanes)):
        distance.append((sum(sum(np.array(gt_lanes[i])**2))**0.5)/len(gt_lanes[i]))
    index = distance.index(min(distance))
    te_value = lane_lcte_adj[index]
    indices = np.where(te_value == 1)[0]
    te_list = []
    for i in range(len(ann_info['traffic_element'])):
        te_ann = ann_info['traffic_element'][i]
        te_id = te_ann['id']
        attribute = te_ann['attribute']
        box = te_ann['points']
        if attribute>=4:
            num+=1
            te_list.append(attribute)
    
    traffic_element_dict[scene_id][timestamp] = te_list
    
te_convert = {4: 'go_straight', 5: 'turn_left', 6: 'turn_right', 7: 'no_left_turn', 8: 'no_right_turn', 9: 'u_turn', 10: 'no_u_turn', 11: 'slight_left', 12: 'slight_right'}

time_span = 5
for scene in traffic_element_dict.keys():
    sequence_info = traffic_element_dict[scene]
    history_list = []
    for timestamp in sequence_info.keys():
        te_list = sequence_info[timestamp]
        history_list.append(te_list)

        longest_list = max(history_list, key=len)
        element_counts = {}
        for element in longest_list:
            if element in element_counts:
                element_counts[element] += 1
            else:
                element_counts[element] = 1
            
        length = len(element_counts)
        Question = 'Has the ego vehicle seen any traffic sign before?'
        if length==0:
            Answer = 'No. There is no traffic sign in the scene.'
        else:
            pass
        if len(history_list)>=5:
            history_list.pop(0)

