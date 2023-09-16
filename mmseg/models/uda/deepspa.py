# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy
import cv2
import fastremap
from scipy import ndimage
from skimage.segmentation import watershed
from torch import einsum

import mmcv
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_revgrad import RevGrad

from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmcv.runner import Sequential

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import UDA, build_segmentor, build_head
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform,
                                                color_jitter, gaussian_blur)
from mmseg.datasets.nuclei_cellpose import result_to_inst
from mmseg.datasets.pipelines.transforms import GetCellPoseMap
from mmseg.datasets.pipelines import transformsgpu
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DeepSpa(UDADecorator):

    def __init__(self, **cfg):
        super(DeepSpa, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']

        # self.debug_img_interval = cfg['debug_img_interval']

    def get_ema_model(self):
        return get_module(self.ema_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas, gt_instance, gt_vec):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_size = img.shape[0]
        dev = img.device
        means, stds = get_mean_std(img_metas, dev)

        # 3. source_img的有监督训练
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, gt_vec=gt_vec, return_last_feat=False)

        loss, log_vars = self._parse_losses(clean_losses)

        loss.backward()

        # if self.local_iter % self.debug_img_interval == 0:
        #     out_dir = os.path.join(self.train_cfg['work_dir'],
        #                            'pseudo_label')
        #     os.makedirs(out_dir, exist_ok=True)
        #     vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
        #     vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
        #
        #     for j in range(batch_size):
        #         rows, cols = 2, 5
        #         fig, axs = plt.subplots(
        #             rows,
        #             cols,
        #             figsize=(3 * cols, 3 * rows),
        #             gridspec_kw={
        #                 'hspace': 0.1,
        #                 'wspace': 0,
        #                 'top': 0.95,
        #                 'bottom': 0,
        #                 'right': 1,
        #                 'left': 0
        #             },
        #         )
        #         subplotimg(axs[0][0], vis_img[j], 'Source Image')
        #         subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
        #         subplotimg(
        #             axs[0][1],
        #             gt_semantic_seg[j],
        #             'Source Seg GT',
        #             cmap='cityscapes')
        #         subplotimg(
        #             axs[1][1],
        #             pseudo_segs[j],
        #             'Target Seg (Pseudo) GT',
        #             cmap='cityscapes')
        #         subplotimg(axs[0][2], gt_instance[j], 'Source Instance GT')
        #         subplotimg(
        #             axs[1][2], pseudo_instances[j], 'Target Instance (Pseudo) GT')
        #         # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
        #         #            cmap="cityscapes")
        #         subplotimg(
        #             axs[0][3], (gt_vec[j, :, :, 0] + 1) * 125, 'Source H gradient GT', vmin=0, vmax=255)
        #         subplotimg(
        #             axs[0][4], (gt_vec[j, :, :, 1] + 1) * 125, 'Source V gradient GT', vmin=0, vmax=255)
        #
        #         subplotimg(
        #             axs[1][3], (pseudo_vecs[j, 0, :, :] + 1) * 125, 'Target H gradient (Pseudo) GT', vmin=0,
        #             vmax=255)
        #         subplotimg(
        #             axs[1][4], (pseudo_vecs[j, 1, :, :] + 1) * 125, 'Target V gradient (Pseudo) GT', vmin=0,
        #             vmax=255)
        #
        #         for ax in axs.flat:
        #             ax.axis('off')
        #         plt.savefig(
        #             os.path.join(out_dir,
        #                          f'{(self.local_iter + 1):06d}_{j}.png'))
        #         plt.close()
        self.local_iter += 1

        return log_vars
