import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
import random



answer_num = []
gt_num = []
failed = 0

five_num, one_num, two_num, four_num, ten_num, twenty_num, thirty_num = 0, 0, 0, 0, 0, 0, 0

with open("lavis/output/ablation/pretrain_num_loc_opt_det/20231126104/result/val_18_vqa_result.json", 'r') as file:
    data = json.load(file)

# with open("utf8_encoded.json", "w", encoding="utf-8") as file:
#     json.dump(data, file, ensure_ascii=False, indent=4)

# with open("utf8_encoded.json", 'r') as file:
#     data = json.load(file)

num = 0
for data_one in data:
    if data_one['gt_answer'][:3] != 'Loc':
        continue
    # if "ago" in data_one['question']:
    #     continue
    # if "later" in data_one['question']:
    #     continue
    answer = data_one['answer']
    gt_answer = data_one['gt_answer']
    # print(answer, gt_answer)
    # input()
    try:
        gt_answer = gt_answer.split('Location: ')[-1].split(' ')
        gt_num = []
        # print(gt_answer)
        gt_num.append(float(gt_answer[0]))
        gt_num.append(float(gt_answer[1]))
        gt_num.append(float(gt_answer[2]))
        if len(gt_num) != 3:
            failed += 1
            print(gt_num)

        # if answer != "":
        #     import pdb; pdb.set_trace()
        label = answer.split("Label: ")[-1].split("_")[0]
        answer = answer.split('Location: ')[-1].split(' ')

        answer_num = []
        answer_num.append(float((answer[0])))
        answer_num.append(float(answer[1]))
        answer_num.append(float(answer[2]))

        dist = np.array(answer_num) - np.array(gt_num)
        distance = (dist[0]**2 + dist[1]**2)**0.5
        if (np.array(gt_num)[0] ** 2 + np.array(gt_num)[1] ** 2) ** 0.5 > 70:
            continue
        num += 1
        # print(answer[4], gt_answer[4])
        # if label != gt_answer[4]:
        #     continue
        
        # print("answer: ", answer_num, "gt: ", gt_num, "distance: ", distance)
    except:
        continue
    
    if distance <= 1:
        one_num += 1
    if distance <= 2:
        two_num += 1
    if distance <= 4:
        four_num += 1
    if distance <= 5:
        five_num += 1
    if distance <= 10:
        ten_num += 1
    if distance <= 20:
        twenty_num += 1
    if distance <= 30:
        thirty_num += 1

    
one_ap = one_num / num
two_ap = two_num / num
four_ap = four_num / num
five_ap = five_num / num
ten_ap = ten_num / num
twenty_ap = twenty_num / num
thirty_ap = thirty_num / num

print(num)
print(failed)

print('one_ap: ', one_ap, 'two_ap: ', two_ap, 'four_ap: ', four_ap, 'five_ap: ', five_ap, 'ten_ap: ', ten_ap, 'twenty_ap: ', twenty_ap, 'thirty_ap: ', thirty_ap)


print("fail rate: ", failed / len(data))