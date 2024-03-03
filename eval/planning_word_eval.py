import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
import random

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

answer_num = []
gt_num = []

with open("lavis/output/ablation/planning_hybrid_pretrain/20231122170/result/val_20_vqa_result.json", 'r') as file:
    data = json.load(file)

with open("utf8_encoded.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

with open("utf8_encoded.json", 'r') as file:
    data = json.load(file)

L2 = 0.0
nums = 0
for data_one in data:
    answer = data_one['answer']
    gt_answer = data_one['gt_answer']
    try:
        answer_num = answer.replace("[", "").replace("]", "").replace("Trajectory:", "").replace(" ", "").split(",")
        assert len(answer_num) == 12, print(data_one['answer'])
        for idx in range(len(answer_num)):
            answer_num[idx] = check_if(answer_num[idx], vocab_to_num)
            if idx % 2 == 0:
                answer_num[idx] = vocab_to_num[answer_num[idx]] - num_threshold
            else:
                answer_num[idx] = vocab_to_num[answer_num[idx]] - num_threshold - 100
        answer_num = np.array([float(num) for num in answer_num]).reshape(-1, 2)
        gt_num = gt_answer.replace("[", "").replace("]", "").replace("Trajectory:", "").replace(" ", "").split(",")
        for idx in range(len(gt_num)):
            gt_num[idx] = check_if(gt_num[idx], vocab_to_num)
            if idx %2 == 0:
                gt_num[idx] = vocab_to_num[gt_num[idx]] - num_threshold
            else:
                gt_num[idx] = vocab_to_num[gt_num[idx]] - num_threshold - 100
        gt_num = np.array([float(num) for num in gt_num]).reshape(-1, 2)
        answer_num = answer_num[[1, 3, 5]]
        gt_num = gt_num[[1, 3, 5]]
        dist = ((np.array(answer_num) - np.array(gt_num)) ** 2).sum(axis=-1)
        distance = (dist**0.5).mean(axis=0)
        L2 += distance
        nums += 1
    except:
        continue
    
L2 = L2 / nums
print(f"nums and total: {nums}, {len(data)}")

print(f"L2 distance: {L2}")

