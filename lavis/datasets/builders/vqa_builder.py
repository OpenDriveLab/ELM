"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.aok_vqa_datasets import AOKVQADataset, AOKVQAEvalDataset
from lavis.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset
from lavis.datasets.datasets.vg_vqa_datasets import VGVQADataset
from lavis.datasets.datasets.gqa_datasets import GQADataset, GQAEvalDataset
from lavis.datasets.datasets.threedvqa_datasets import ThreeDVQADataset, ThreeDVQAEvalDataset
from lavis.datasets.datasets.NLQ_datasets import NLQDataset, NLQDatasetEvalDataset
from lavis.datasets.datasets.narration_datasets import NARRATIONDataset, NARRATIONDatasetEvalDataset
from lavis.datasets.datasets.combine_datasets import COMBINEDataset, COMBINEDatasetEvalDataset
from lavis.datasets.datasets.llama_datasets import LLAMADataset, LLAMADatasetEvalDataset
from lavis.datasets.datasets.lr_narration_datasets import LRNARRATIONDataset, LRNARRATIONDatasetEvalDataset
from lavis.datasets.datasets.traffic_flag_datasets import TRAFFICDataset, TRAFFICDatasetEvalDataset
from lavis.datasets.datasets.emdmulti_datasets import EMDMULTIDataset, EMDMULTIDatasetEvalDataset


@registry.register_builder("coco_vqa")
class COCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset
    eval_dataset_cls = COCOVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa.yaml",
        "eval": "configs/datasets/coco/eval_vqa.yaml",
    }


@registry.register_builder("vg_vqa")
class VGVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VGVQADataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_vqa.yaml"}


@registry.register_builder("ok_vqa")
class OKVQABuilder(COCOVQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults.yaml",
    }


@registry.register_builder("aok_vqa")
class AOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset
    eval_dataset_cls = AOKVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa/defaults.yaml"}


@registry.register_builder("gqa")
class GQABuilder(BaseDatasetBuilder):
    train_dataset_cls = GQADataset
    eval_dataset_cls = GQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/defaults.yaml",
        "balanced_val": "configs/datasets/gqa/balanced_val.yaml",
        "balanced_testdev": "configs/datasets/gqa/balanced_testdev.yaml",
    }

@registry.register_builder("3d_vqa")
class ThreeDVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = ThreeDVQADataset
    eval_dataset_cls = ThreeDVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/3dvqa/defaults.yaml"}

@registry.register_builder("nlq")
class NLQBuilder(BaseDatasetBuilder):
    train_dataset_cls = NLQDataset
    eval_dataset_cls = NLQDatasetEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/nlq/defaults.yaml"}

@registry.register_builder("narration")
class NARRATIONBuilder(BaseDatasetBuilder):
    train_dataset_cls = NARRATIONDataset
    eval_dataset_cls = NARRATIONDatasetEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/narration/defaults.yaml"}

@registry.register_builder("combine")
class COMEBINEBuilder(BaseDatasetBuilder):
    train_dataset_cls = COMBINEDataset
    eval_dataset_cls = COMBINEDatasetEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/combine/defaults.yaml"}

@registry.register_builder("llama")
class LLAMABuilder(BaseDatasetBuilder):
    train_dataset_cls = LLAMADataset
    eval_dataset_cls = LLAMADatasetEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/llama/defaults.yaml"}

@registry.register_builder("lrnarration")
class LRNARRATIONBuilder(BaseDatasetBuilder):
    train_dataset_cls = LRNARRATIONDataset
    eval_dataset_cls = LRNARRATIONDatasetEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/lrnarration/defaults.yaml"}

@registry.register_builder("traffic")
class TRAFFICBuilder(BaseDatasetBuilder):
    train_dataset_cls = TRAFFICDataset
    eval_dataset_cls = TRAFFICDatasetEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/traffic/defaults.yaml"}

@registry.register_builder("emdmulti")
class EMDMULTIBuilder(BaseDatasetBuilder):
    train_dataset_cls = EMDMULTIDataset
    eval_dataset_cls = EMDMULTIDatasetEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/emdmulti/defaults.yaml"}