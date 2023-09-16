from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class TamperDataset(CustomDataset):
    CLASSES = ('normal', 'tamper')
    palette = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(TamperDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=None,
            **kwargs)
