import sys

import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
from lavis.processors import BlipQuestionProcessor
import json
import random


# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# /cpfs01/shared/opendrivelab/opendrivelab_hdd/linyan/VLM/models/loc_det_elm.pth
state_dict = torch.load('lavis/output/ablation/planning_num_pretrain/20231125013/checkpoint_47.pth')['model']
# for key in list(state_dict.keys()):
#     state_dict[key.replace('base_model.model.', '')] = state_dict.pop(key)

# Other available models:
# 
text_processer = BlipQuestionProcessor()
model_ours, vis_processors, _ = load_model_and_preprocess(
    name="blip2_vqa_t5_emdmulti", model_type="pretrain_flant5xl", is_eval=True, device=device
)
model_ours.load_state_dict(state_dict, strict=False)


count = 0
time = 0

# model.eval()
with open('data/embodied/planning_val.json', 'r') as f:
    val_dataset = json.load(f)

import pdb; pdb.set_trace()

random.shuffle(val_dataset)
# val_dataset = val_dataset[-15000:]

for data in val_dataset[:1000]:
    # if count == 50:
    #     break
    # question = data['conversations'][0]['value']
    question = 'Find the 3D position in the scene of the 2D pixel at <c, CAM_FRONT, 800.0, 450.0>.'
    # gt_answer = data['conversations'][1]['value']
    # for img in img_path:
    question = text_processer(question)

    img_path = data['img_path']#[-1] if isinstance(data['image'], list) else data['image']
    # caption = data['caption']
    if isinstance(data['img_path'], list):
        imgs = []
        timestamp = torch.zeros((1, len(data['img_path'])))
        for idx, img in enumerate(img_path):
            raw_image = Image.open(img).convert('RGB')   
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            imgs.append(image)
            if gt_answer.replace(' ', '_') in img:
                print(gt_answer.replace(' ', '_'), img)
                timestamp[0,idx] = 1
                if idx >= 1:
                    timestamp[0,idx-1] = 1
                if idx < len(data['img_path']) - 1:
                    timestamp[0,idx+1] = 1
        
        image = torch.cat(imgs, dim=0).unsqueeze(0).to(device)
    else:
        raw_image = Image.open(img_path).convert('RGB')   
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    output = model_ours.predict_answers({"vfeats": image, "questions": question})[0]

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    time += elapsed

    count += 1
    print(img_path)
    print('Q ', question)
    print('Ours ', output)

print(time/count)

