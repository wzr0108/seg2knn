# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

from .compose import Compose
from .formating import (Collect, ImageToTensor, ToDataContainer, ToTensor,
                        Transpose, to_tensor)
from .loading import LoadAnnotations, LoadImageFromFile, LoadAnnotationsNpy, LoadNpyFromFile, LoadAnnotationsMat
from .test_time_aug import MultiScaleFlipAug
from .transforms import (CLAHE, AdjustGamma, Normalize, Pad,
                         PhotoMetricDistortion, RandomCrop, RandomFlip,
                         RandomRotate, Rerange, Resize, RGB2Gray, SegRescale,
                         GetHVMap, GetCellPoseMap, MyAug, MyRandomRotate, ResizeToMultiple, RandomRotate90, Albu,
                         KeepWeakImg)
from . import transformsgpu

# from .tamper_tranforms import RandomRemove, RandomCopyMove

__all__ = [
    'Compose', 'to_tensor', 'ToTensor', 'ImageToTensor', 'ToDataContainer',
    'Transpose', 'Collect', 'LoadAnnotations', 'LoadImageFromFile',
    'MultiScaleFlipAug', 'Resize', 'RandomFlip', 'Pad', 'RandomCrop',
    'Normalize', 'SegRescale', 'PhotoMetricDistortion', 'RandomRotate',
    'AdjustGamma', 'CLAHE', 'Rerange', 'RGB2Gray', 'LoadAnnotationsNpy',
    'GetHVMap', "GetCellPoseMap", 'transformsgpu', 'MyAug', 'MyRandomRotate', 'ResizeToMultiple', 'RandomRotate90',
    'Albu', "KeepWeakImg", "LoadNpyFromFile", 'LoadAnnotationsMat'

]
