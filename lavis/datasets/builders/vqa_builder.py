"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.elm_datasets import ELMDataset, ELMDatasetEvalDataset
from lavis.datasets.datasets.lr_narration_datasets import LRNARRATIONDataset, LRNARRATIONDatasetEvalDataset


@registry.register_builder("elm")
class ELMBuilder(BaseDatasetBuilder):
    train_dataset_cls = ELMDataset
    eval_dataset_cls = ELMDatasetEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/elm/defaults.yaml"}

@registry.register_builder("lr_narration")
class IMGNARRATIONBuilder(BaseDatasetBuilder):
    train_dataset_cls = LRNARRATIONDataset
    eval_dataset_cls = LRNARRATIONDatasetEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/lr_narration/defaults.yaml"}