import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
import random
import language_evaluation

data_root = 'lavis/output/ablation/learnable_ego4d'
evaluator = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])


log_name = '20240131061'

with open("keep.json", 'r') as file:
    mask = np.array(json.load(file))

for i in range(0, 8):
    epoch_name = 'val_{}_vqa_result.json'.format(i)

    answer_list = []
    gt_list = []

    five_num, one_num, two_num, four_num = 0, 0, 0, 0
    try:
        # print(epoch_name)
        with open(os.path.join(data_root,log_name,'result',epoch_name), 'r') as file:
            data = json.load(file)
    except:
        continue
    # data = [data[i] for i, value in enumerate(mask) if value == 1]
    for data_one in data:
        # if "seconds before in the history" in data_one['question']: # 6976 Moment Recap
        #     continue
        # if "What happened between" in data_one['question']: # 6977 Event query
        #     continue
        # if "What will happen in the next" in data_one['question']: # 6895 Action Forcasting
        #     continue
        question = data_one['question']
        answer = data_one['answer']
        gt_answer = data_one['gt_answer']
        
        # if "A: " not in answer:
        #     answer = "A: " + answer
        
        # if answer == "A: ":
        #     continue

        # if "A:  " in answer:
        #     continue

        # if len(answer.split(" ")) < 4 and len(gt_answer.split(" ")) > 8:
        #     continue
        # if np.linalg.norm(len(answer.split(" ")) - len(gt_answer.split(" "))) > 1:
        #     continue

    

        # answer = answer.replace('A: ', '')
        # gt_answer = gt_answer.replace('A: ', '')
        # print(answer, gt_answer)

        # event1 = question.split("What happened between '")[1].split("' and '")[0]
        # event2 = question.split("What happened between '")[1].split("' and '")[1]

        # if gt_answer == event1 or gt_answer == event2:
        #     print(f"{gt_answer}, {event1}, {event2}")
        #     continue

        answer.replace('</s>','').replace('<pad>','')
        answer_list.append(answer)
        gt_list.append(gt_answer)
    print(len(answer_list))
    results_gen = evaluator.run_evaluation(
        answer_list, gt_list
    )

    results_gen_dict = {
        f"val/{k}": v for k, v in results_gen.items()
    }

    accuracy = sum([1 if gt == pred else 0 for gt, pred in zip(gt_list, answer_list)]) / len(gt_list)

    ap = sum([v for k, v in results_gen_dict.items()]) / len(results_gen_dict)
        

    print(i, 'one_ap: ', accuracy, ap)
    print(results_gen_dict)

# answer_list = []
# gt_list = []

# with open('lavis/output/BLIP2/drivelm/QA.txt', 'r') as f:
#     data = f.readlines()
#     for i in range(len(data)):
#         if "Answer:" in data[i]:
#             gt_list.append(data[i].split('Answer:')[1].strip())
#         if "Generated:" in data[i]:
#             answer_list.append(data[i].split('Generated:')[1].strip())
#     assert len(answer_list) == len(gt_list)
#     print(len(answer_list))

#     results_gen = evaluator.run_evaluation(
#         answer_list, gt_list
#     )

#     results_gen_dict = {
#         f"val/{k}": v for k, v in results_gen.items()
#     }

#     accuracy = sum([1 if gt == pred else 0 for gt, pred in zip(gt_list, answer_list)]) / len(gt_list)

#     ap = sum([v for k, v in results_gen_dict.items()]) / len(results_gen_dict)
        
        

#     print(i, 'one_ap: ', accuracy, ap)
#     print(results_gen_dict)
