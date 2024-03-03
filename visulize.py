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


if __name__ == "__main__":

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

    answer_num = []
    gt_num = []
    failed = 0

    five_num, one_num, two_num, four_num = 0, 0, 0, 0

    with open("lavis/output/BLIP2/pretrain/20231028043/result/val_19_vqa_result.json", 'r') as file:
        data = json.load(file)

    with open("utf8_encoded.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    with open("utf8_encoded.json", 'r') as file:
        data = json.load(file)
    
    gt_points = []
    pred_points = []
    for data_one in data:
        answer = data_one['answer']
        gt_answer = data_one['gt_answer']
        gt_answer = check_if(gt_answer, vocab_to_num)
        # print(answer, gt_answer)
        # input()
        try:
            gt_answer = gt_answer.split('Location: ')[-1].split(' ')
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
            if gt_num[1] > 70 or answer_num[0] > 70 or answer_num[1] > 70:
                continue
            # print("answer: ", answer_num, "gt: ", gt_num, "distance: ", distance)
        except:
            continue

        gt_points.append(gt_num)
        pred_points.append(answer_num)
    
    
    # gt_points = gt_points[::100]
    # pred_points = pred_points[::100]

    # # visulize it
    # pred_x, pred_y, _ = zip(*pred_points)
    # plt.scatter(pred_x, pred_y, c='red', label='pred', s=4, alpha=0.05)

    # # gt_x, gt_y, _ = zip(*gt_points)
    # # plt.scatter(gt_x, gt_y, c='green', label='gt', s=4, alpha=0.5)


    # # for g, p in zip(gt_points, pred_points):
    # #     plt.plot([p[0], g[0]], [p[1], g[1]], c='black', alpha=0.2, linewidth=1)
    
    # plt.savefig('2.png')



    with open("data/embodied/BOXQA_val.json") as f:
        data = json.load(f)
    
    foreground = []
    background = []
    boxqa = []
    for name in data.keys():
        info = data[name]['key_frame']
        for time in info.keys():
            if info[time] is None:
                continue
            if "Pretrain Fore Loc" in info[time].keys():
                fore = info[time]["Pretrain Fore Loc"]["a"]
            else:
                fore = []
            if "Pretrain Back Loc" in info[time].keys():
                back = info[time]["Pretrain Back Loc"]["a"]
            else:
                back = []
            if "BOX QA" in info[time].keys():
                box = info[time]["BOX QA"]["a"]
            else:
                box = []

            for f in fore:
                foreground.append([float(i) for i in f.split("A: ")[-1].split(", ")])
            for b in back:
                background.append([float(i) for i in b.split("A: ")[-1].split(", ")])
            for x in box:
                boxqa.append([float(i) for i in x.split("A: ")[-1].split(", ")])

    alldata = foreground + background + boxqa
    print(f"The number of foreground is {len(foreground)}")
    print(f"The number of background is {len(background)}")
    print(f"The number of boxqa is {len(boxqa)}")
    
    near = 0 
    far = 0
    for fore in boxqa:
        dist = (fore[0] ** 2 + fore[1] ** 2) ** 0.5
        if dist < 30:
            near += 1
        else:
            far += 1
    print("The ratio is ", near / far)
    
    
    keep = []
    for data in alldata:
        if data[1] > 70 or data[0] > 70 or data[2] > 70:
            continue
        keep.append(data)
    keep = keep[::300]
    gt_x, gt_y, _ = zip(*keep)
    plt.scatter(gt_x, gt_y, c='green', label='gt', s=2, alpha=0.05)
    plt.savefig('1.png')