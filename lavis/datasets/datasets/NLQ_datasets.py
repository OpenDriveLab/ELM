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

def preprocess_target(s_labels, e_labels, h_labels):
    # process targets
    B = len(s_labels)
    s_labels_string = [i.numpy() for i in s_labels] # [8]
    e_labels_string = [i.numpy() for i in e_labels] # [8]

    h_indexes = []
    for i in range(B):
        h_indexes.append(np.where(h_labels[i].numpy() == 1)[0])

    gt_output_text = []
    for i in range(B):
        gt_output_text.append(str(s_labels_string[i]//4))
        # gt_output_nums.append(str(s_labels_string[i]) + ',' + str(e_labels_string[i]) + ',' + str(h_indexes[i]))

    return gt_output_text


class NLQDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        data_path = "data/ego4d/v2/nlq_official_v2_omnivore_video_fp16_official_128_bert.pkl"
        data = load_pickle(data_path)
        
        self.train_data = data["val_set"]
        self.video_features = load_video_features(
            os.path.join("data/ego4d/v2", "nlq_official_v2_omnivore_video_fp16", "official"), 128)
        self.keys = list(self.video_features.keys())[0]
    

    def __getitem__(self, index):
        record = self.train_data[index]
        video_feature = self.video_features[record["vid"]]#[self.keys] #[record["vid"]]
        s_ind, e_ind = int(record["s_ind"]), int(record["e_ind"])
        word_ids = record["w_ids"]
        char_ids = record.get("c_ids", None)
        question = record["query"]

        return {
            "record": record,
            "video_feature": video_feature,
            "word_ids": word_ids,
            "char_ids": char_ids,
            "s_ind": s_ind,
            "e_ind": e_ind,
            "question": question,
        }

    def __len__(self):
        return len(self.train_data)

    def collater(self, samples):
        # merge samples into a list for each key
        records = [s["record"] for s in samples]
        video_features = [s["video_feature"] for s in samples]
        word_ids = [s["word_ids"] for s in samples]
        char_ids = [s["char_ids"] for s in samples]
        s_inds = [s["s_ind"] for s in samples]
        e_inds = [s["e_ind"] for s in samples]
        question = [s["question"] for s in samples]

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

        # process targets
        gt_output_text = preprocess_target(s_labels, e_labels, h_labels)

        return {
            "records": records,
            "vfeats": vfeats,
            "vfeat_lens": vfeat_lens,
            "word_ids": word_ids,
            "char_ids": char_ids,
            "s_labels": s_labels,
            "e_labels": e_labels,
            "h_labels": h_labels,
            "question": question,
            "gt_output_text": gt_output_text,
        }


class NLQDatasetEvalDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        data_path = "data/ego4d/v2/nlq_official_v2_omnivore_video_fp16_official_128_bert.pkl"
        data = load_pickle(data_path)
        
        self.test_data = data["val_set"][:10]
        self.video_features = load_video_features(
            os.path.join("data/ego4d/v2", "nlq_official_v2_omnivore_video_fp16", "official"), 128)

    def __getitem__(self, index):
        record = self.test_data[index]
        video_feature = self.video_features[record["vid"]]
        s_ind, e_ind = int(record["s_ind"]), int(record["e_ind"])
        word_ids = record["w_ids"]
        char_ids = record.get("c_ids", None)
        question = record["query"]

        return {
            "record": record,
            "video_feature": video_feature,
            "word_ids": word_ids,
            "char_ids": char_ids,
            "s_ind": s_ind,
            "e_ind": e_ind,
            "question": question,
        }

    def __len__(self):
        return len(self.test_data)

    def collater(self, samples):
        # merge samples into a list for each key
        records = [s["record"] for s in samples]
        video_features = [s["video_feature"] for s in samples]
        word_ids = [s["word_ids"] for s in samples]
        char_ids = [s["char_ids"] for s in samples]
        s_inds = [s["s_ind"] for s in samples]
        e_inds = [s["e_ind"] for s in samples]
        question = [s["question"] for s in samples]

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

        # process targets
        gt_output_text = preprocess_target(s_labels, e_labels, h_labels)

        return {
            "records": records,
            "vfeats": vfeats,
            "vfeat_lens": vfeat_lens,
            "word_ids": word_ids,
            "char_ids": char_ids,
            "s_labels": s_labels,
            "e_labels": e_labels,
            "h_labels": h_labels,
            "question": question,
            "gt_output_text": gt_output_text,
        }
