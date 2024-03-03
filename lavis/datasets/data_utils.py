"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import glob
import json
import pickle

import gzip
import logging
import os
import random as rnd
import tarfile
import zipfile

import decord
import webdataset as wds
import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset, ChainDataset
from decord import VideoReader
from lavis.common.registry import registry
from lavis.datasets.datasets.base_dataset import ConcatDataset
from tqdm import tqdm
import torch.nn as nn

decord.bridge.set_bridge("torch")
MAX_INT = registry.get("MAX_INT")


def load_video(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform"):
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start, end, vlen / n_frms).astype(int)
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    frms = vr.get_batch(indices).permute(3, 0, 1, 2).float()  # (C, T, H, W)

    return frms


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    # TODO fp16 support

    return samples


def reorg_datasets_by_split(datasets):
    """
    Organizes datasets by split.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by name.

    Returns:
        Dict of datasets by split {split_name: List[Datasets]}.
    """
    # if len(datasets) == 1:
    #     return datasets[list(datasets.keys())[0]]
    # else:
    reorg_datasets = dict()

    # reorganize by split
    for _, dataset in datasets.items():
        for split_name, dataset_split in dataset.items():
            if split_name not in reorg_datasets:
                reorg_datasets[split_name] = [dataset_split]
            else:
                reorg_datasets[split_name].append(dataset_split)

    return reorg_datasets


def concat_datasets(datasets):
    """
    Concatenates multiple datasets into a single dataset.

    It supports may-style datasets and DataPipeline from WebDataset. Currently, does not support
    generic IterableDataset because it requires creating separate samplers.

    Now only supports conctenating training datasets and assuming validation and testing
    have only a single dataset. This is because metrics should not be computed on the concatenated
    datasets.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by split.

    Returns:
        Dict of concatenated datasets by split, "train" is the concatenation of multiple datasets,
        "val" and "test" remain the same.

        If the input training datasets contain both map-style and DataPipeline datasets, returns
        a tuple, where the first element is a concatenated map-style dataset and the second
        element is a chained DataPipeline dataset.

    """
    # concatenate datasets in the same split
    for split_name in datasets:
        if split_name != "train":
            assert (
                len(datasets[split_name]) == 1
            ), "Do not support multiple {} datasets.".format(split_name)
            datasets[split_name] = datasets[split_name][0]
        else:
            iterable_datasets, map_datasets = [], []
            for dataset in datasets[split_name]:
                if isinstance(dataset, wds.DataPipeline):
                    logging.info(
                        "Dataset {} is IterableDataset, can't be concatenated.".format(
                            dataset
                        )
                    )
                    iterable_datasets.append(dataset)
                elif isinstance(dataset, IterableDataset):
                    raise NotImplementedError(
                        "Do not support concatenation of generic IterableDataset."
                    )
                else:
                    map_datasets.append(dataset)

            # if len(iterable_datasets) > 0:
            # concatenate map-style datasets and iterable-style datasets separately
            chained_datasets = (
                ChainDataset(iterable_datasets) if len(iterable_datasets) > 0 else None
            )
            concat_datasets = (
                ConcatDataset(map_datasets) if len(map_datasets) > 0 else None
            )

            train_datasets = concat_datasets, chained_datasets
            train_datasets = tuple([x for x in train_datasets if x is not None])
            train_datasets = (
                train_datasets[0] if len(train_datasets) == 1 else train_datasets
            )

            datasets[split_name] = train_datasets

    return datasets


def extract_archive(from_path, to_path=None, overwrite=False):
    """Extract archive.

    Args:
        from_path: the path of the archive.
        to_path: the root path of the extracted files (directory of from_path)
        overwrite: overwrite existing files (False)

    Returns:
        List of paths to extracted files even if not overwritten.

    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> from_path = './validation.tar.gz'
        >>> to_path = './'
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']

    """

    if to_path is None:
        to_path = os.path.dirname(from_path)

    if from_path.endswith((".tar.gz", ".tgz")):
        logging.info("Opening tar file {} to {}.".format(from_path, to_path))
        with tarfile.open(from_path, "r") as tar:
            files = []
            for file_ in tqdm(tar):
                file_path = os.path.join(to_path, file_.name)
                if file_.isfile():
                    files.append(file_path)
                    if os.path.exists(file_path):
                        logging.info("{} already extracted.".format(file_path))
                        if not overwrite:
                            continue
                tar.extract(file_, to_path)
            logging.info("Finished extracting tar file {}.".format(from_path))
            return files

    elif from_path.endswith(".zip"):
        assert zipfile.is_zipfile(from_path), from_path
        logging.info("Opening zip file {} to {}.".format(from_path, to_path))
        with zipfile.ZipFile(from_path, "r") as zfile:
            files = []
            for file_ in tqdm(zfile.namelist()):
                file_path = os.path.join(to_path, file_)
                files.append(file_path)
                if os.path.exists(file_path):
                    logging.info("{} already extracted.".format(file_path))
                    if not overwrite:
                        continue
                zfile.extract(file_, to_path)
        files = [f for f in files if os.path.isfile(f)]
        logging.info("Finished extracting zip file {}.".format(from_path))
        return files

    elif from_path.endswith(".gz"):
        logging.info("Opening gz file {} to {}.".format(from_path, to_path))
        default_block_size = 65536
        filename = from_path[:-3]
        files = [filename]
        with gzip.open(from_path, "rb") as gzfile, open(filename, "wb") as d_file:
            while True:
                block = gzfile.read(default_block_size)
                if not block:
                    break
                else:
                    d_file.write(block)
            d_file.write(block)
        logging.info("Finished extracting gz file {}.".format(from_path))
        return files

    else:
        raise NotImplementedError(
            "We currently only support tar.gz, .tgz, .gz and zip achives."
        )


def save_frames_grid(img_array, out_path):
    import torch
    from PIL import Image
    from torchvision.utils import make_grid

    if len(img_array.shape) == 3:
        img_array = img_array.unsqueeze(0)
    elif len(img_array.shape) == 5:
        b, t, c, h, w = img_array.shape
        img_array = img_array.view(-1, c, h, w)
    elif len(img_array.shape) == 4:
        pass
    else:
        raise NotImplementedError(
            "Supports only (b,t,c,h,w)-shaped inputs. First two dimensions can be ignored."
        )

    assert img_array.shape[1] == 3, "Exepcting input shape of (H, W, 3), i.e. RGB-only."

    grid = make_grid(img_array)
    ndarr = grid.permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    img = Image.fromarray(ndarr)

    img.save(out_path)




def load_json(filename):
    with open(filename, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, mode="w", encoding="utf-8") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_lines(filename):
    with open(filename, mode="r", encoding="utf-8") as f:
        return [e.strip("\n") for e in f.readlines()]


def save_lines(data, filename):
    with open(filename, mode="w", encoding="utf-8") as f:
        f.write("\n".join(data))


def load_pickle(filename):
    with open(filename, mode="rb") as handle:
        data = pickle.load(handle)
        return data


def save_pickle(data, filename):
    with open(filename, mode="wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_video_features(root, max_position_length):
    video_features = dict()
    extension = "*.pt"

    filenames = glob.glob(os.path.join(root, extension))
    for filename in tqdm(filenames, total=len(filenames), desc="load video features"):
        video_id = filename.split("/")[-1].split(".")[0]
        feature = torch.load(filename).to(torch.float32).cpu().numpy()
        if max_position_length is None:
            video_features[video_id] = feature
        else:
            new_feature = visual_feature_sampling(
                feature, max_num_clips=max_position_length
            )
            video_features[video_id] = new_feature
            # if new_feature.shape[0] != feature.shape[0]:
            #    print(f"Reduced: {feature.shape[0]} --> {new_feature.shape[0]}")
    return video_features


def visual_feature_sampling(visual_feature, max_num_clips):
    num_clips = visual_feature.shape[0]
    if num_clips <= max_num_clips:
        return visual_feature
    idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_visual_feature = []
    for i in range(max_num_clips):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_visual_feature.append(np.mean(visual_feature[s_idx:e_idx], axis=0))
        else:
            new_visual_feature.append(visual_feature[s_idx])
    new_visual_feature = np.asarray(new_visual_feature)
    return new_visual_feature


def compute_overlap(pred, gt):
    # check format
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    pred = pred if pred_is_list else [pred]
    gt = gt if gt_is_list else [gt]
    # compute overlap
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(1e-12, union_right - union_left)
    overlap = 1.0 * inter / union
    # reformat output
    overlap = overlap if gt_is_list else overlap[:, 0]
    overlap = overlap if pred_is_list else overlap[0]
    return overlap


def time_to_index(start_time, end_time, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) / float(num_units) * duration
    e_times = (
        np.arange(1, num_units + 1).astype(np.float32) / float(num_units) * duration
    )
    candidates = np.stack(
        [
            np.repeat(s_times[:, None], repeats=num_units, axis=1),
            np.repeat(e_times[None, :], repeats=num_units, axis=0),
        ],
        axis=2,
    ).reshape((-1, 2))
    overlaps = compute_overlap(candidates.tolist(), [start_time, end_time]).reshape(
        num_units, num_units
    )
    start_index = np.argmax(overlaps) // num_units
    end_index = np.argmax(overlaps) % num_units
    return start_index, end_index, overlaps


def index_to_time(start_index, end_index, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = (
        np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    )
    start_time = s_times[start_index]
    end_time = e_times[end_index]
    return start_time, end_time


def pad_seq(sequences, pad_tok=None, max_length=None):
    if pad_tok is None:
        pad_tok = 0  # 0: "PAD" for words and chars, "PAD" for tags
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length


def pad_char_seq(sequences, max_length=None, max_length_2=None):
    sequence_padded, sequence_length = [], []
    if max_length is None:
        max_length = max(map(lambda x: len(x), sequences))
    if max_length_2 is None:
        max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    for seq in sequences:
        sp, sl = pad_seq(seq, max_length=max_length_2)
        sequence_padded.append(sp)
        sequence_length.append(sl)
    sequence_padded, _ = pad_seq(
        sequence_padded, pad_tok=[0] * max_length_2, max_length=max_length
    )
    sequence_length, _ = pad_seq(sequence_length, max_length=max_length)
    return sequence_padded, sequence_length


def pad_video_seq(sequences, max_length=None):
    if max_length is None:
        max_length = max([vfeat.shape[0] for vfeat in sequences])
    feature_length = sequences[0].shape[1]
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        add_length = max_length - seq.shape[0]
        sequence_length.append(seq.shape[0])
        if add_length > 0:
            add_feature = np.zeros(shape=[add_length, feature_length], dtype=np.float32)
            seq_ = np.concatenate([seq, add_feature], axis=0)
        else:
            seq_ = seq
        sequence_padded.append(seq_)
    return sequence_padded, sequence_length


def load_video_features(root, max_position_length):
    video_features = dict()
    extension = "*.pt"

    filenames = glob.glob(os.path.join(root, extension))
    for filename in tqdm(filenames, total=len(filenames), desc="load video features"):
        video_id = filename.split("/")[-1].split(".")[0]
        feature = torch.load(filename).to(torch.float32).cpu().numpy()
        if max_position_length is None:
            video_features[video_id] = feature
        else:
            new_feature = visual_feature_sampling(
                feature, max_num_clips=max_position_length
            )
            video_features[video_id] = new_feature
            # if new_feature.shape[0] != feature.shape[0]:
            #    print(f"Reduced: {feature.shape[0]} --> {new_feature.shape[0]}")
    return video_features


def load_pickle(filename):
    with open(filename, mode="rb") as handle:
        data = pickle.load(handle)
        return data


def convert_length_to_mask(lengths):
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(
        lengths.size()[0], max_len
    ) < lengths.unsqueeze(1)
    mask = mask.float()
    return mask


def compute_loss(scores, labels, mask, epsilon=1e-12):
    labels = labels.type(torch.float32)
    weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
    loss_per_location = nn.BCEWithLogitsLoss(reduction="none")(scores, labels)
    loss_per_location = loss_per_location * weights
    mask = mask.type(torch.float32)
    loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + epsilon)
    return loss
