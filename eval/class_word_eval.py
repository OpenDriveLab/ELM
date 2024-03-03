import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
import random
import pickle


def check_if(word, voc):
    if word in voc:
        return word
    else:
        return "▁" + word

vocab_to_num = {}
num_threshold = 30000
with open('data/vocab.txt', 'r') as file:
    for line_number, line_content in enumerate(file, 1):
        line_content = line_content.strip()
        if line_number>=(num_threshold-1000):
            vocab_to_num[line_content] = line_number

loc_x_bin = [-17.18, -13.36, -11.08, -9.17, -7.17, -5.64, -4.07, -2.91, -1.3, 0.61, 2.27, 3.25, 4.11, 5.06, 6.14, 7.45, 9.02, 11.21, 15.17] # 20
loc_y_bin = [9.69, 12.54, 15.09, 17.31, 19.48, 21.55, 23.56, 25.62, 27.56, 29.47, 31.42, 33.54, 35.68, 38.12, 40.66, 43.4, 46.42, 50.04, 54.52] # 20
loc_z_bin =  [-1.31, -0.94, -0.52, 0.03]    # 5
dim_l_bin =  [0.46, 0.56, 0.64, 0.73, 0.92, 1.75, 1.83, 1.89, 1.95, 2.01, 2.08, 2.21, 2.56, 3.05] # 15
dim_h_bin = [0.54, 0.82, 4.36, 4.82] # 5
dim_w_bin = [0.96, 1.12, 1.44, 1.57, 1.66, 1.75, 1.82, 1.93, 2.22]  # 10
yaw_bin = [-4.36, -3.30, -3.13, -2.95, -1.6, -1.16, -0.05, 0.04, 1.04]  # 10
velo_x_bin = [-0.05, 0.0, 0.05] # 4
velo_y_bin = [-0.07, 0.0, 0.09] # 4


label_mapping = {
    'noise': 'noise',
    'animal': 'noise',  # 没有对应的labels_16类别
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.personal_mobility': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.stroller': 'pedestrian',
    'human.pedestrian.wheelchair': 'pedestrian',
    'movable_object.barrier': 'barrier',
    'movable_object.debris': 'noise',  # 没有对应的labels_16类别
    'movable_object.pushable_pullable': 'noise',  # 没有对应的labels_16类别
    'movable_object.trafficcone': 'traffic_cone',
    'static_object.bicycle_rack': 'noise',  # 没有对应的labels_16类别
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'noise',  # 没有对应的labels_16类别
    'vehicle.emergency.police': 'noise',  # 没有对应的labels_16类别
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck',
    'flat.driveable_surface': 'driveable_surface',
    'flat.other': 'other_flat',
    'flat.sidewalk': 'sidewalk',
    'flat.terrain': 'terrain',
    'static.manmade': 'manmade',
    'static.other': 'noise',  # 没有对应的labels_16类别
    'static.vegetation': 'vegetation',
    'vehicle.ego': 'noise',  # 没有对应的labels_16类别

    "car": "car",
    "pedestrian": "pedestrian",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "truck": "truck",
    "bus": "bus",
    "trailer": "trailer",
    "construction_vehicle": "construction_vehicle",
    "traffic_cone": "traffic_cone",
    "barrier": "barrier",
    "noise": "noise"
}


answer_num = []
gt_num = []
failed = 0

five_num, one_num, two_num, four_num, ten_num, twenty_num, thirty_num = 0, 0, 0, 0, 0, 0, 0

with open("lavis/output/ablation/det_replace_words_multiframe/20240125141/result/val_6_vqa_result.json", 'r') as file:
    data = json.load(file)

with open("utf8_encoded.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

with open("utf8_encoded.json", 'r') as file:
    data = json.load(file)


num = 0
statics = {}
statics["car"] = {}
statics["pedestrian"] = {}
statics["bicycle"] = {}
statics["motorcycle"] = {}
statics["truck"] = {}
statics["bus"] = {}
statics["trailer"] = {}
statics["construction_vehicle"] = {}
statics["traffic_cone"] = {}
statics["barrier"] = {}
statics["noise"] = {}

nums_all = {"car": 0, "pedestrian": 0, "bicycle": 0, "motorcycle": 0, "truck": 0, "bus": 0, "trailer": 0, "construction_vehicle": 0, "traffic_cone": 0, "barrier": 0, "noise": 0}


for cls_id, data_one in enumerate(data):
    answer = data_one['answer']
    gt_answer = data_one['gt_answer']
    gt_answer = check_if(gt_answer, vocab_to_num)
    # import pdb; pdb.set_trace()

    try:
        gt_answer = gt_answer.split('Location: ')[-1].split(' ')

        class_id = gt_answer[4]
        gt_answer = gt_answer[:3]
        for idx in range(len(gt_answer)):
            gt_answer[idx] = check_if(gt_answer[idx], vocab_to_num)
        gt_num = []
        if gt_answer[0] in vocab_to_num.keys():
            gt_num.append(vocab_to_num[gt_answer[0]] - num_threshold)
        if gt_answer[1] in vocab_to_num.keys():
            gt_num.append(vocab_to_num[gt_answer[1]] - num_threshold-100)
        if gt_answer[2] in vocab_to_num.keys():
            gt_num.append(vocab_to_num[gt_answer[2]] - num_threshold-300)
        if len(gt_num) != 3:
            failed += 1
            print(gt_num)

        answer = answer.split('Location: ')[-1].split(' ')
        #import pdb; pdb.set_trace()
        for idx in range(len(answer)):
            answer[idx] = check_if(answer[idx], vocab_to_num)

        answer_num = []
        if answer[0] in vocab_to_num.keys():
            answer_num.append(vocab_to_num[answer[0]] - num_threshold)
        if answer[1] in vocab_to_num.keys():
            answer_num.append(vocab_to_num[answer[1]] - num_threshold-100)
        if answer[2] in vocab_to_num.keys():
            answer_num.append(vocab_to_num[answer[2]] - num_threshold-300)

        dist = np.array(answer_num) - np.array(gt_num)
        distance = (dist[0]**2 + dist[1]**2)**0.5
        if (np.array(gt_num)[0] ** 2 + np.array(gt_num)[1] ** 2) ** 0.5 > 100:
            continue
        num += 1
        
    except:
        continue

    nums_all[label_mapping[class_id]] += 1
    if distance <= 1:
        if answer[4][1:] == class_id:
            statics[label_mapping[class_id]]["one"] = statics[label_mapping[class_id]].get("one", 0) + 1
    if distance <= 2:
        if answer[4][1:] == class_id:
            statics[label_mapping[class_id]]["two"] = statics[label_mapping[class_id]].get("two", 0) + 1
    if distance <= 4:
        if answer[4][1:] == class_id:
            statics[label_mapping[class_id]]["four"] = statics[label_mapping[class_id]].get("four", 0) + 1
    if distance <= 5:
        if answer[4][1:] == class_id:
            statics[label_mapping[class_id]]["five"] = statics[label_mapping[class_id]].get("five", 0) + 1
    # if distance <= 10:
    #     ten_num += 1

for cate in statics.keys():
    for key in statics[cate].keys():
        statics[cate][key] = statics[cate][key] / nums_all[cate]


precision = 0.0
length = 0
for cate in statics.keys():
    if "one" in statics[cate]:
        length += 1
        precision += statics[cate]["one"]

precision = precision / length

print(precision)
print(statics)
print(nums_all)