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
from mmseg.models.utils import ProtoEstimator
from mmseg.models.losses.contrastive_loss import contrast_preparations


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


def sigmoid_ramp_up(iter, max_iter):
    """

    Args:
        iter: current iteration
        max_iter: maximum number of iterations to perform the rampup

    Returns:
        returns 1 if iter >= max_iter
        returns [0,1] incrementally from 0 to max_iters if iter < max_iter

    """
    if iter >= max_iter:
        return 1
    else:
        return np.exp(- 5 * (1 - float(iter) / float(max_iter)) ** 2)


def entropy_loss(v, mask=None):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()

    loss_image = torch.mul(v, torch.log2(v + 1e-30))
    loss_image = torch.sum(loss_image, dim=1)
    if mask is not None:
        loss_image = mask.float().squeeze(1) * loss_image
        percentage_valid_points = torch.mean(mask.float())
    else:
        percentage_valid_points = 1.0

    return -torch.sum(loss_image) / (n * h * w * np.log2(c) * percentage_valid_points)


def augmentationTransform(parameters, data=None, target=None, probs=None, jitter_vale=0.4, min_sigma=0.2, max_sigma=2.,
                          ignore_label=255):
    """

    Args:
        parameters: dictionary with the augmentation configuration
        data: BxCxWxH input data to augment
        target: BxWxH labels to augment
        probs: BxWxH probability map to augment
        jitter_vale:  jitter augmentation value
        min_sigma: min sigma value for blur
        max_sigma: max sigma value for blur
        ignore_label: value for ignore class


    Returns:
            augmented data, target, probs
    """
    assert ((data is not None) or (target is not None))
    if "Mix" in parameters:
        data, target, probs = transformsgpu.mix(mask=parameters["Mix"], data=data, target=target, probs=probs)

    if "RandomScaleCrop" in parameters:
        data, target, probs = transformsgpu.random_scale_crop(scale=parameters["RandomScaleCrop"], data=data,
                                                              target=target, probs=probs,
                                                              ignore_label=ignore_label
                                                              )
    if "flip" in parameters:
        data, target, probs = transformsgpu.flip(flip=parameters["flip"], data=data, target=target, probs=probs,
                                                 )

    if "ColorJitter" in parameters:
        data, target, probs = transformsgpu.colorJitter(colorJitter=parameters["ColorJitter"], data=data, target=target,
                                                        probs=probs, s=jitter_vale)
    if "GaussianBlur" in parameters:
        data, target, probs = transformsgpu.gaussian_blur(blur=parameters["GaussianBlur"], data=data, target=target,
                                                          probs=probs, min_sigma=min_sigma, max_sigma=max_sigma)

    if "Grayscale" in parameters:
        data, target, probs = transformsgpu.grayscale(grayscale=parameters["Grayscale"], data=data, target=target,
                                                      probs=probs)
    if "Solarize" in parameters:
        data, target, probs = transformsgpu.solarize(solarize=parameters["Solarize"], data=data, target=target,
                                                     probs=probs)

    return data, target, probs


def augment_samples(images, labels, probs, do_classmix, batch_size, ignore_label, weak=False):
    """
    Perform data augmentation

    Args:
        images: BxCxWxH images to augment
        labels:  BxWxH labels to augment
        probs:  BxWxH probability maps to augment
        do_classmix: whether to apply classmix augmentation
        batch_size: batch size
        ignore_label: ignore class value
        weak: whether to perform weak or strong augmentation

    Returns:
        augmented data, augmented labels, augmented probs

    """

    if do_classmix:
        raise NotImplementedError
        # ClassMix: Get mask for image A
        # for image_i in range(batch_size):  # for each image
        #     classes = torch.unique(labels[image_i])  # get unique classes in pseudolabel A
        #     nclasses = classes.shape[0]
        #
        #     # remove ignore class
        #     if ignore_label in classes and len(classes) > 1 and nclasses > 1:
        #         classes = classes[classes != ignore_label]
        #         nclasses = nclasses - 1
        #
        #     if dataset == 'pascal_voc':  # if voc dataaset, remove class 0, background
        #         if 0 in classes and len(classes) > 1 and nclasses > 1:
        #             classes = classes[classes != 0]
        #             nclasses = nclasses - 1
        #
        #     # pick half of the classes randomly
        #     classes = (classes[torch.Tensor(
        #         np.random.choice(nclasses, int(((nclasses - nclasses % 2) / 2) + 1), replace=False)).long()]).cuda()
        #
        #     # acumulate masks
        #     if image_i == 0:
        #         MixMask = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()
        #     else:
        #         MixMask = torch.cat(
        #             (MixMask, transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()))
        #
        # params = {"Mix": MixMask}
    else:
        params = {}

    if weak:
        params["flip"] = random.random() < 0.5
        params["ColorJitter"] = random.random() < 0.2
        params["GaussianBlur"] = random.random() < 0.
        params["Grayscale"] = random.random() < 0.0
        params["Solarize"] = random.random() < 0.0
        if random.random() < 0.5:
            scale = random.uniform(0.75, 1.75)
        else:
            scale = 1
        params["RandomScaleCrop"] = scale

        # Apply strong augmentations to unlabeled images
        image_aug, labels_aug, probs_aug = augmentationTransform(params,
                                                                 data=images, target=labels,
                                                                 probs=probs, jitter_vale=0.125,
                                                                 min_sigma=0.1, max_sigma=1.5,
                                                                 ignore_label=ignore_label,
                                                                 )
    else:
        params["flip"] = random.random() < 0.5
        params["ColorJitter"] = random.random() < 0.8
        params["GaussianBlur"] = random.random() < 0.2
        params["Grayscale"] = random.random() < 0.0
        params["Solarize"] = random.random() < 0.0
        if random.random() < 0.80:
            scale = random.uniform(0.75, 1.75)
        else:
            scale = 1
        params["RandomScaleCrop"] = scale

        # Apply strong augmentations to unlabeled images
        image_aug, labels_aug, probs_aug = augmentationTransform(params,
                                                                 data=images, target=labels,
                                                                 probs=probs, jitter_vale=0.25,
                                                                 min_sigma=0.1, max_sigma=1.5,
                                                                 ignore_label=ignore_label,
                                                                 )

    return image_aug, labels_aug, probs_aug, params


@UDA.register_module()
class SePiCo(UDADecorator):
    def __init__(self, **cfg):
        super(SePiCo, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.ramp_up_iter = cfg['ramp_up_iter']

        self.alpha = cfg['alpha']
        self.debug_img_interval = cfg['debug_img_interval']

        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        # 用于计算伪标签损失
        self.unlbl_loss1 = nn.CrossEntropyLoss(reduction="none", ignore_index=255)

        # 对比学习参数
        self.proj_dim = cfg['model']['auxiliary_head']['channels']
        self.contrast_mode = cfg['model']['auxiliary_head']['input_transform']
        self.calc_layers = cfg['model']['auxiliary_head']['in_index']
        self.num_classes = cfg['model']['auxiliary_head']['num_classes']
        self.enable_avg_pool = cfg['model']['auxiliary_head']['loss_decode']['use_avg_pool']
        self.scale_min_ratio = cfg['model']['auxiliary_head']['loss_decode']['scale_min_ratio']
        self.memory_length = cfg.get('memory_length', 0)  # 0 means no memory bank

        self.ignore_index = 255
        self.start_distribution_iter = cfg['start_distribution_iter']

        # init distribution
        if self.contrast_mode == 'multiple_select':
            self.feat_distributions = {}
            for idx in range(len(self.calc_layers)):
                self.feat_distributions[idx] = ProtoEstimator(dim=self.proj_dim, class_num=self.num_classes,
                                                              memory_length=self.memory_length)
        else:  # 'resize_concat' or None
            self.feat_distributions = ProtoEstimator(dim=self.proj_dim, class_num=self.num_classes,
                                                     memory_length=1024)

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
                      target_img_metas, **kwargs):
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
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
        # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
        # assert not _params_equal(self.get_ema_model(), self.get_model())
        # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)

        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        # 1. 给target_img生成伪标签，使用ema_model
        with torch.no_grad():
            target_logits = self.get_ema_model().encode_decode(target_img, target_img_metas)  # [bs, 9, H, W]
            target_logits = target_logits[:, -5:, :, :]  # # [bs, 5, H, W]
            target_softmax = torch.softmax(target_logits.detach(), dim=1)
            target_max_probs, pseudo_labels = torch.max(target_softmax, dim=1)

        # 2. 对target_img进行数据增强，然后使用伪标签训练
        # semiseg进行了两次增强，这里只进行一次
        # class_mix没搞懂，先设置为False
        # 把pseudo_vec和pseudo_seg合并，增强后再分开
        # 4个变量分别 为增强后的图片、对应的伪标签、伪标签置信度、增强的参数(后面没用到)
        target_img_aug, \
        pseudo_label_aug, target_max_prob_aug, target_aug_params = augment_samples(target_img,
                                                                                   pseudo_labels,
                                                                                   probs=target_max_probs,
                                                                                   do_classmix=False,
                                                                                   batch_size=len(
                                                                                       target_img),
                                                                                   ignore_label=self.ignore_index,
                                                                                   weak=False,
                                                                                   )

        pixel_wise_weight = sigmoid_ramp_up(self.local_iter, self.ramp_up_iter) * torch.ones(
            target_max_prob_aug.shape)
        pixel_wise_weight = pixel_wise_weight.to(dev)
        pixel_wise_weight = pixel_wise_weight * torch.pow(target_max_prob_aug.detach(), 6)

        target_losses = self.get_model().forward_train(
            target_img_aug, target_img_metas, pseudo_label_aug.unsqueeze(1), seg_weight=pixel_wise_weight
        )

        target_losses["pseudo_weight"] = pixel_wise_weight.mean()
        target_loss, target_log_vars = self._parse_losses(target_losses)
        log_vars.update(add_prefix(target_log_vars, 'tgt'))
        target_loss.backward()

        # 3. 更新特征分布
        src_feat = self.get_ema_model().extract_auxiliary_feat(img)
        mean = {}
        covariance = {}
        bank = {}
        if self.contrast_mode == 'multiple_select':
            for idx in range(len(self.calc_layers)):
                feat, mask = contrast_preparations(src_feat[idx], gt_semantic_seg, self.enable_avg_pool,
                                                   self.scale_min_ratio, self.num_classes, self.ignore_index)
                self.feat_distributions[idx].update_proto(features=feat.detach(), labels=mask)
                mean[idx] = self.feat_distributions[idx].Ave
                covariance[idx] = self.feat_distributions[idx].CoVariance
                bank[idx] = self.feat_distributions[idx].MemoryBank
        else:
            # feat [n,1024]
            # mask [n,]
            feat, mask = contrast_preparations(src_feat, gt_semantic_seg, self.enable_avg_pool,
                                               self.scale_min_ratio, self.num_classes, self.ignore_index)
            self.feat_distributions.update_proto(features=feat.detach(), labels=mask)
            mean = self.feat_distributions.Ave
            covariance = self.feat_distributions.CoVariance
            bank = self.feat_distributions.MemoryBank  # list of deque

        # 4. source_img的有监督训练和对比学习
        if self.local_iter >= self.start_distribution_iter:
            mode = "all"
        else:
            mode = "dec"

        gt_vec = kwargs["gt_vec"]
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, gt_vec=gt_vec, mode="dec",
            mean=mean, covariance=covariance, bank=bank
        )
        source_loss, source_log_vars = self._parse_losses(clean_losses)
        log_vars.update(add_prefix(source_log_vars, "src"))
        source_loss.backward()

        # 5. target的对比学习
        if self.local_iter >= self.start_distribution_iter:
            target_losses = self.get_model().forward_train(
                target_img_aug, target_img_metas, pseudo_label_aug.unsqueeze(1), mode="aux", mean=mean,
                covariance=covariance, bank=bank
            )
            target_loss, target_log_vars = self._parse_losses(target_losses)
            log_vars.update(add_prefix(target_log_vars, "tgt"))
            target_loss.backward()

        self.local_iter += 1

        return log_vars
