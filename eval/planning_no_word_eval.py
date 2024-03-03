import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
import random

data_root = 'lavis/output/ablation/planning_waymo_ood'
log_name = '20240130025'
epoch_name = 'val_11_vqa_result.json' # 'val_46_vqa_result.json'


answer_num = []
gt_num = []

five_num, one_num, two_num, four_num = 0, 0, 0, 0
with open(os.path.join(data_root,log_name,'result',epoch_name), 'r') as file:
    data = json.load(file)

L2 = 0.0
FDE = 0.0
nums = 0
mask = [0] * 6019
pred_trajectory = [None] * 6019
for data_one in data:
    answer = data_one['answer']
    gt_answer = data_one['gt_answer']
    idx = int(gt_answer.split("__")[-1])
    gt_answer = gt_answer.split("__")[0]
    try:
        answer_num = answer.replace("[", "").replace("]", "").replace("Trajectory:", "").replace(" ", "").split(",")
        # assert len(answer_num) == 12#, print(data_one['answer'])
        answer_num = np.array([float(num) for num in answer_num]).reshape(-1, 2)
        gt_num = gt_answer.replace("[", "").replace("]", "").replace("Trajectory:", "").replace(" ", "").split(",")
        assert len(gt_num) == 12, print(".............................")
        gt_num = np.array([float(num) for num in gt_num]).reshape(-1, 2)
        # answer_num = answer_num[[1, 3, 5]]
        # gt_num = gt_num[[1, 3, 5]]
        # if "forward" not in data_one['question']:
        #     continue
        dist = ((np.array(answer_num[[1, 3, 5]]) - np.array(gt_num[[1, 3, 5]])) ** 2).sum(axis=-1)
        distance = (dist**0.5).mean(axis=0)
        # if distance > 5:
        #     print(data_one['answer'], data_one['gt_answer'])
        if np.linalg.norm(gt_num[-1]) > 50:
            continue
        L2 += distance
        FDE += (((answer_num[[-1]] - gt_num[[-1]]) ** 2).sum(axis=-1) ** 0.5).mean(axis=0)
        nums += 1
        pred_trajectory[idx] = answer_num.reshape(1, -1).tolist()
        mask[idx] = 1
    except:
        mask[idx] = 0
        pred_trajectory[idx] = None
        continue
    
L2 = L2 / nums
print(f"nums and total: {nums}, {len(data)}")

print(f"L2 distance: {L2}")
print(f"FDE distance: {FDE / nums}")

# # save 
# print(len(pred_trajectory))
# with open(os.path.join(data_root,log_name,'result',epoch_name.replace(".json", "_pred.json")), 'w') as file:
#     json.dump(pred_trajectory, file)

# j = 0

# with open(os.path.join(data_root,log_name,'result',epoch_name.replace(".json", "_mask.json")), 'w') as file:
#     json.dump(mask, file)