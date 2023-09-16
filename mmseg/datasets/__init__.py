# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional datasets

from .acdc import ACDCDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .cityscapes import CityscapesDataset
from .custom import CustomDataset
from .dark_zurich import DarkZurichDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .gta import GTADataset
from .synthia import SynthiaDataset
from .uda_dataset import UDADataset
from .nuclei_cellpose import NucleiCellPoseDataset, result_to_inst
from .tamper import TamperDataset
from .teeth import TeethDataset
from .ade import ADE20KDataset
from .nuclei_hv2 import NucleiHV2Dataset

__all__ = [
    'CustomDataset',
    'build_dataloader',
    'ConcatDataset',
    'RepeatDataset',
    'DATASETS',
    'build_dataset',
    'PIPELINES',
    'CityscapesDataset',
    'GTADataset',
    'SynthiaDataset',
    'UDADataset',
    'ACDCDataset',
    'DarkZurichDataset',
    'NucleiCellPoseDataset',
    'result_to_inst',
    'TamperDataset',
    'TeethDataset',
    'ADE20KDataset',
    'NucleiHV2Dataset'
]
