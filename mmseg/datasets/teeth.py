import tempfile
import os.path as osp
from PIL import Image

import mmcv
import numpy as np

from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class TeethDataset(CustomDataset):
    CLASSES = ('background', 'foreground')
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, **kwargs):
        super(TeethDataset, self).__init__(**kwargs)
        # 读取连续书数据时，判断每个nii切片最大和最小的索引
        self.nii_max_min = {}
        for info in self.img_infos:
            filename = info["filename"]
            nii_name = filename[:-8]
            idx = int(filename[-7:-4])
            if self.nii_max_min.get(f"{nii_name}_min", None) is None:
                self.nii_max_min[f"{nii_name}_min"] = idx
            else:
                self.nii_max_min[f"{nii_name}_min"] = min(self.nii_max_min[f"{nii_name}_min"], idx)

            if self.nii_max_min.get(f"{nii_name}_max", None) is None:
                self.nii_max_min[f"{nii_name}_max"] = idx
            else:
                self.nii_max_min[f"{nii_name}_max"] = max(self.nii_max_min[f"{nii_name}_max"], idx)

    def format_results(self, results, imgfile_prefix=None, to_255=False):
        """Format the results into dir

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_255:

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None

        mmcv.mkdir_or_exist(imgfile_prefix)
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            if to_255:
                result = result * 255
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]
            png_filename = osp.join(imgfile_prefix, f'{basename}.png')
            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            prog_bar.update()

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info, nii_max_min=self.nii_max_min)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info, nii_max_min=self.nii_max_min)
        self.pre_pipeline(results)
        return self.pipeline(results)
