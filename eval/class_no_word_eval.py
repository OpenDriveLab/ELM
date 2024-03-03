import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
import random
import pickle


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

with open("lavis/output/ablation/pretrain_num_loc_opt_det/20231126104/result/val_18_vqa_result.json", 'r') as file:
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

    try:
        # import pdb; pdb.set_trace()
        gt_answer = gt_answer.split('Location: ')[-1].split(' ')

        class_id = gt_answer[4]
        # import pdb; pdb.set_trace()
        gt_answer = gt_answer[:3]
        gt_num = []
        gt_num.append(float(gt_answer[0]))
        gt_num.append(float(gt_answer[1]))
        gt_num.append(float(gt_answer[2]))
        if len(gt_num) != 3:
            failed += 1
            print(gt_num)

        label = answer.split("Label: ")[-1].split("_")[0]
        answer = answer.split('Location: ')[-1].split(' ')[:3]
        answer.append("Label:")
        answer.append(label)

        answer_num = []
        answer_num.append(float((answer[0])))
        answer_num.append(float(answer[1]))
        answer_num.append(float(answer[2]))

        dist = np.array(answer_num) - np.array(gt_num)
        distance = (dist[0]**2 + dist[1]**2)**0.5
        if (np.array(gt_num)[0] ** 2 + np.array(gt_num)[1] ** 2) ** 0.5 > 100:
            continue
        num += 1
        
    except:
        continue

    nums_all[label_mapping[class_id]] += 1
    if distance <= 1:
        if answer[4] == class_id:
            statics[label_mapping[class_id]]["one"] = statics[label_mapping[class_id]].get("one", 0) + 1
    if distance <= 2:
        if answer[4] == class_id:
            statics[label_mapping[class_id]]["two"] = statics[label_mapping[class_id]].get("two", 0) + 1
    if distance <= 4:
        if answer[4] == class_id:
            statics[label_mapping[class_id]]["four"] = statics[label_mapping[class_id]].get("four", 0) + 1
    if distance <= 5:
        if answer[4] == class_id:
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