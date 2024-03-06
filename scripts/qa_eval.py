# Description: This script is used to evaluate the QA model on the validation set.
# Usage: python qa_eval.py data_root log_name
# You may need to install language-evaluation package following the instructions in https://github.com/bckim92/language-evaluation
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import language_evaluation

data_root = sys.argv[1]
evaluator = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])

log_name = sys.argv[2]


for i in range(0,20):
    epoch_name = 'val_{}_vqa_result.json'.format(i)

    answer_list = []
    gt_list = []

    five_num, one_num, two_num, four_num = 0, 0, 0, 0
    try:
        with open(os.path.join(data_root,log_name,'result',epoch_name), 'r') as file:
            data = json.load(file)
    except:
        continue
    for data_one in data:
        question = data_one['question']
        answer = data_one['answer']
        gt_answer = data_one['gt_answer']

        answer_list.append(answer)
        gt_list.append(gt_answer)
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