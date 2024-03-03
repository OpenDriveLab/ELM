import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
import random

data_root = 'lavis/output/BLIP2/hist_box_ours'
# log_name = '20231001065' # 0.122
# log_name = '20231002075' # 0.375
# log_name = '20231003133' # 0.140
# log_name = '20231003114' # 0.393
# log_name = '20231004045' # 0.357 0.378
# log_name = '20231005043' #
# log_name = '20231005064' #
# log_name = '20231006081' #
# log_name = '20231007105' # 0.0059
# log_name = '20231007131' # 0.374
log_name = '20231024090' # 0.384
# log_name = '20231004071' # 0.151
# log_name = '20231002094' # 0.156
epoch_name = 'val_1_vqa_result.json'

loc_x_bin = [-17.18, -13.36, -11.08, -9.17, -7.17, -5.64, -4.07, -2.91, -1.3, 0.61, 2.27, 3.25, 4.11, 5.06, 6.14, 7.45, 9.02, 11.21, 15.17] # 20
loc_y_bin = [9.69, 12.54, 15.09, 17.31, 19.48, 21.55, 23.56, 25.62, 27.56, 29.47, 31.42, 33.54, 35.68, 38.12, 40.66, 43.4, 46.42, 50.04, 54.52] # 20
loc_z_bin =  [-1.31, -0.94, -0.52, 0.03]    # 5
dim_l_bin =  [0.46, 0.56, 0.64, 0.73, 0.92, 1.75, 1.83, 1.89, 1.95, 2.01, 2.08, 2.21, 2.56, 3.05] # 15
dim_h_bin = [0.54, 0.82, 4.36, 4.82] # 5
dim_w_bin = [0.96, 1.12, 1.44, 1.57, 1.66, 1.75, 1.82, 1.93, 2.22]  # 10
yaw_bin = [-4.36, -3.30, -3.13, -2.95, -1.6, -1.16, -0.05, 0.04, 1.04]  # 10
velo_x_bin = [-0.05, 0.0, 0.05] # 4
velo_y_bin = [-0.07, 0.0, 0.09] # 4

answer_num = []
gt_num = []

five_num, one_num, two_num, four_num = 0, 0, 0, 0
with open(os.path.join(data_root,log_name,'result',epoch_name), 'r') as file:
    data = json.load(file)
for data_one in data:
    answer = data_one['answer']
    gt_answer = data_one['gt_answer']
    # print(answer, gt_answer)
    # input()
    try:
        if '[' not in gt_answer and ']' not in gt_answer:
                continue
        if 'A:' in gt_answer:
                continue
        answer_num = re.findall(r'-?\d+', answer)
        answer_num = [int(num) for num in answer_num]
        # answer_num = [loc_x_bin[(answer_num[0])*2], loc_y_bin[(answer_num[1]-10)*2], loc_z_bin[answer_num[2]-20]]
        # answer_num = [answer_num[0]*3.6, answer_num[1]*5, answer_num[2]*0.3]
        gt_num = re.findall(r'-?\d+', gt_answer)
        gt_num = [int(num) for num in gt_num]
        # gt_num = [loc_x_bin[(gt_num[0])*2], loc_y_bin[(gt_num[1]-10)*2], loc_z_bin[gt_num[2]-20]]
        # gt_num = [gt_num[0]*3.6, gt_num[1]*5, gt_num[2]*0.3]
        dist = np.array(answer_num) - np.array(gt_num)
        distance = (dist[0]**2 + dist[1]**2)**0.5
    except:
        continue
    
    if distance < 1:
        one_num += 1
    if distance < 2:
        two_num += 1
    if distance < 4:
        four_num += 1
    if distance < 5:
        five_num += 1
    
one_ap = one_num / len(data)
two_ap = two_num / len(data)
four_ap = four_num / len(data)
five_ap = five_num / len(data)

print('one_ap: ', one_ap, 'two_ap: ', two_ap, 'four_ap: ', four_ap, 'five_ap: ', five_ap)

