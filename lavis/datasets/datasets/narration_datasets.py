"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import torch
import numpy as np
import pickle

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from lavis.datasets.data_utils import load_video_features, pad_video_seq, pad_seq, pad_char_seq, load_pickle
from nuscenes.nuscenes import NuScenes
import re
from collections import OrderedDict
import tqdm

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answer": "; ".join(ann["answers"]),
                "pc_feat": sample["pc_feat"],
                "pc": sample["pc"],
            }
        )

class NARRATIONDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        data_path = "data/ego4d/v2/nlq_official_v2_omnivore_video_fp16_official_128_bert.pkl"
        data = load_pickle(data_path)

        self.train_data = data["train_set"]
        self.video_features = load_video_features(
            os.path.join("data/ego4d/v2", "nlq_official_v2_omnivore_video_fp16", "official"), 128)
        self.keys = list(self.video_features.keys())[0]

        # process answers
        narration_file_path = "data/ego4d/v2/annotations/narration.json"
        narration_data = load_json_file(narration_file_path)

        nlq_file_path = "data/ego4d/v2/annotations/nlq_train.json"
        nlq_data = load_json_file(nlq_file_path)

        video_clip_dict = {}

        for video in nlq_data["videos"]:
            video_uid = video['video_uid']
            for clip in video['clips']:
                clip_uid = clip['clip_uid']
                clip_s_time = clip['video_start_sec']
                clip_e_time = clip['video_end_sec']
                video_clip_dict[clip_uid] = [video_uid, clip_s_time, clip_e_time]
        
        self.data_infos = []
        self.questions = []
        self.answers = []
        for i in tqdm.tqdm(range(len(self.train_data))):
            record = self.train_data[i]
            # print(record)
            # input()
            clip_id, start_time, end_time = record["vid"], record["s_time"], record["e_time"]
            s_ind, e_ind = record["s_ind"], record["e_ind"]
            question = record["query"]
            video_id = video_clip_dict[clip_id][0]
            clip_s_time, clip_e_time = video_clip_dict[clip_id][1], video_clip_dict[clip_id][2]
            video_clip = narration_data[video_id]
            if "narration_pass_1" in video_clip:
                narrations = video_clip["narration_pass_1"]["narrations"]
            else:
                continue

            for narration in narrations[:len(narrations)//2]:
                self.questions.append(narration['narration_text'])
                self.answers.append("Yes")
                self.data_infos.append(self.train_data[i])
            for narration in narrations[len(narrations)//2:]:
                self.questions.append(narration['narration_text'])
                self.answers.append("No")
                self.data_infos.append(self.train_data[i])
    

    def __getitem__(self, index):
        record = self.data_infos[index]
        video_feature = self.video_features[record["vid"]]#[self.keys] #[record["vid"]]
        s_ind, e_ind = int(record["s_ind"]), int(record["e_ind"])
        word_ids = record["w_ids"]
        char_ids = record.get("c_ids", None)
        question = self.questions[index]
        answer = self.answers[index]

        return {
            "record": record,
            "video_feature": video_feature,
            "word_ids": word_ids,
            "char_ids": char_ids,
            "s_ind": s_ind,
            "e_ind": e_ind,
            "question": question,
            "answer": answer,
        }

    def __len__(self):
        return len(self.data_infos)

    def collater(self, samples):
        # merge samples into a list for each key
        records = [s["record"] for s in samples]
        video_features = [s["video_feature"] for s in samples]
        word_ids = [s["word_ids"] for s in samples]
        char_ids = [s["char_ids"] for s in samples]
        s_inds = [s["s_ind"] for s in samples]
        e_inds = [s["e_ind"] for s in samples]
        questions = [s["question"] for s in samples]
        answers = [s["answer"] for s in samples]

        # process video features
        vfeats, vfeat_lens = pad_video_seq(video_features)
        vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
        vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )

        if not isinstance(word_ids[0], list):
            pad_input_ids, _ = pad_seq([ii["input_ids"] for ii in word_ids])
            pad_attention_mask, _ = pad_seq([ii["attention_mask"] for ii in word_ids])
            pad_token_type_ids, _ = pad_seq([ii["token_type_ids"] for ii in word_ids])
            word_ids = {
                "input_ids": torch.LongTensor(pad_input_ids),
                "attention_mask": torch.LongTensor(pad_attention_mask),
                "token_type_ids": torch.LongTensor(pad_token_type_ids),
            }
            char_ids = None
        else:
            # process word ids
            word_ids, _ = pad_seq(word_ids)
            word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
            # process char ids
            char_ids, _ = pad_char_seq(char_ids)
            char_ids = np.asarray(
                char_ids, dtype=np.int32
            )  # (batch_size, w_seq_len, c_seq_len)
            word_ids = torch.tensor(word_ids, dtype=torch.int64)
            char_ids = torch.tensor(char_ids, dtype=torch.int64)

        s_labels = np.asarray(s_inds, dtype=np.int64)
        e_labels = np.asarray(e_inds, dtype=np.int64)

        max_len = np.max(vfeat_lens)
        batch_size = vfeat_lens.shape[0]
        h_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
        extend = 0.1
        for idx in range(batch_size):
            st, et = s_inds[idx], e_inds[idx]
            cur_max_len = vfeat_lens[idx]
            extend_len = round(extend * float(et - st + 1))
            if extend_len > 0:
                st_ = max(0, st - extend_len)
                et_ = min(et + extend_len, cur_max_len - 1)
                h_labels[idx][st_ : (et_ + 1)] = 1
            else:
                h_labels[idx][st : (et + 1)] = 1

        vfeats = torch.tensor(vfeats, dtype=torch.float32)
        vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
        s_labels = torch.tensor(s_labels, dtype=torch.int64)
        e_labels = torch.tensor(e_labels, dtype=torch.int64)
        h_labels = torch.tensor(h_labels, dtype=torch.int64)

        return {
            "records": records,
            "vfeats": vfeats,
            "vfeat_lens": vfeat_lens,
            "word_ids": word_ids,
            "char_ids": char_ids,
            "s_labels": s_labels,
            "e_labels": e_labels,
            "h_labels": h_labels,
            "questions": questions,
            "answers": answers,
        }


class NARRATIONDatasetEvalDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        data_path = "data/ego4d/v2/nlq_official_v2_omnivore_video_fp16_official_128_bert.pkl"
        data = load_pickle(data_path)

        self.val_data = data["val_set"]
        self.video_features = load_video_features(
            os.path.join("data/ego4d/v2", "nlq_official_v2_omnivore_video_fp16", "official"), 128)
        self.keys = list(self.video_features.keys())[0]

        # process answers
        narration_file_path = "data/ego4d/v2/annotations/narration.json"
        narration_data = load_json_file(narration_file_path)

        nlq_file_path = "data/ego4d/v2/annotations/nlq_val.json"
        nlq_data = load_json_file(nlq_file_path)

        video_clip_dict = {}

        for video in nlq_data["videos"]:
            video_uid = video['video_uid']
            for clip in video['clips']:
                clip_uid = clip['clip_uid']
                clip_s_time = clip['video_start_sec']
                clip_e_time = clip['video_end_sec']
                video_clip_dict[clip_uid] = [video_uid, clip_s_time, clip_e_time]
        
        self.questions = []
        self.answers = []
        self.data_infos = []
        for i in tqdm.tqdm(range(len(self.val_data))):
            record = self.val_data[i]
            # print(record)
            # input()
            clip_id, start_time, end_time = record["vid"], record["s_time"], record["e_time"]
            s_ind, e_ind = record["s_ind"], record["e_ind"]
            question = record["query"]
            video_id = video_clip_dict[clip_id][0]
            clip_s_time, clip_e_time = video_clip_dict[clip_id][1], video_clip_dict[clip_id][2]
            video_clip = narration_data[video_id]
            if "narration_pass_1" in video_clip:
                narrations = video_clip["narration_pass_1"]["narrations"]
            else:
                continue

            for narration in narrations[:len(narrations)//2]:
                self.questions.append(narration['narration_text'])
                self.answers.append("Yes")
                self.data_infos.append(self.val_data[i])
            for narration in narrations[len(narrations)//2:]:
                self.questions.append(narration['narration_text'])
                self.answers.append("No")
                self.data_infos.append(self.val_data[i])
    

    def __getitem__(self, index):
        record = self.data_infos[index]
        video_feature = self.video_features[record["vid"]]#[self.keys] #[record["vid"]]
        s_ind, e_ind = int(record["s_ind"]), int(record["e_ind"])
        word_ids = record["w_ids"]
        char_ids = record.get("c_ids", None)
        question = self.questions[index]
        answer = self.answers[index]

        return {
            "record": record,
            "video_feature": video_feature,
            "word_ids": word_ids,
            "char_ids": char_ids,
            "s_ind": s_ind,
            "e_ind": e_ind,
            "question": question,
            "answer": answer,
        }

    def __len__(self):
        return len(self.data_infos)

    def collater(self, samples):
        # merge samples into a list for each key
        records = [s["record"] for s in samples]
        video_features = [s["video_feature"] for s in samples]
        word_ids = [s["word_ids"] for s in samples]
        char_ids = [s["char_ids"] for s in samples]
        s_inds = [s["s_ind"] for s in samples]
        e_inds = [s["e_ind"] for s in samples]
        questions = [s["question"] for s in samples]
        answers = [s["answer"] for s in samples]

        # process video features
        vfeats, vfeat_lens = pad_video_seq(video_features)
        vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
        vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )

        if not isinstance(word_ids[0], list):
            pad_input_ids, _ = pad_seq([ii["input_ids"] for ii in word_ids])
            pad_attention_mask, _ = pad_seq([ii["attention_mask"] for ii in word_ids])
            pad_token_type_ids, _ = pad_seq([ii["token_type_ids"] for ii in word_ids])
            word_ids = {
                "input_ids": torch.LongTensor(pad_input_ids),
                "attention_mask": torch.LongTensor(pad_attention_mask),
                "token_type_ids": torch.LongTensor(pad_token_type_ids),
            }
            char_ids = None
        else:
            # process word ids
            word_ids, _ = pad_seq(word_ids)
            word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
            # process char ids
            char_ids, _ = pad_char_seq(char_ids)
            char_ids = np.asarray(
                char_ids, dtype=np.int32
            )  # (batch_size, w_seq_len, c_seq_len)
            word_ids = torch.tensor(word_ids, dtype=torch.int64)
            char_ids = torch.tensor(char_ids, dtype=torch.int64)

        s_labels = np.asarray(s_inds, dtype=np.int64)
        e_labels = np.asarray(e_inds, dtype=np.int64)

        max_len = np.max(vfeat_lens)
        batch_size = vfeat_lens.shape[0]
        h_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
        extend = 0.1
        for idx in range(batch_size):
            st, et = s_inds[idx], e_inds[idx]
            cur_max_len = vfeat_lens[idx]
            extend_len = round(extend * float(et - st + 1))
            if extend_len > 0:
                st_ = max(0, st - extend_len)
                et_ = min(et + extend_len, cur_max_len - 1)
                h_labels[idx][st_ : (et_ + 1)] = 1
            else:
                h_labels[idx][st : (et + 1)] = 1

        vfeats = torch.tensor(vfeats, dtype=torch.float32)
        vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
        s_labels = torch.tensor(s_labels, dtype=torch.int64)
        e_labels = torch.tensor(e_labels, dtype=torch.int64)
        h_labels = torch.tensor(h_labels, dtype=torch.int64)

        return {
            "records": records,
            "vfeats": vfeats,
            "vfeat_lens": vfeat_lens,
            "word_ids": word_ids,
            "char_ids": char_ids,
            "s_labels": s_labels,
            "e_labels": e_labels,
            "h_labels": h_labels,
            "questions": questions,
            "answers": answers,
        }
