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
                          ignore_label=255, vec=None):
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
        vec: Bx2xWxH cellpose的gt

    Returns:
            augmented data, target, probs
    """
    assert ((data is not None) or (target is not None))
    if "Mix" in parameters:
        data, target, probs = transformsgpu.mix(mask=parameters["Mix"], data=data, target=target, probs=probs)

    if "RandomScaleCrop" in parameters:
        data, target, probs, vec = transformsgpu.random_scale_crop(scale=parameters["RandomScaleCrop"], data=data,
                                                                   target=target, probs=probs,
                                                                   ignore_label=ignore_label,
                                                                   vec=vec)
    if "flip" in parameters:
        data, target, probs, vec = transformsgpu.flip(flip=parameters["flip"], data=data, target=target, probs=probs,
                                                      vec=vec)

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

    return data, target, probs, vec


def augment_samples(images, labels, probs, do_classmix, batch_size, ignore_label, weak=False, vec=None):
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
        vec: cellpose的gt

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
        image_aug, labels_aug, probs_aug, vec = augmentationTransform(params,
                                                                      data=images, target=labels,
                                                                      probs=probs, jitter_vale=0.125,
                                                                      min_sigma=0.1, max_sigma=1.5,
                                                                      ignore_label=ignore_label,
                                                                      vec=vec)
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
        image_aug, labels_aug, probs_aug, vec = augmentationTransform(params,
                                                                      data=images, target=labels,
                                                                      probs=probs, jitter_vale=0.25,
                                                                      min_sigma=0.1, max_sigma=1.5,
                                                                      ignore_label=ignore_label,
                                                                      vec=vec)

    return image_aug, labels_aug, probs_aug, params, vec


def contrastive_class_to_class_learned_memory(model, features, class_labels, num_classes, memory, device):
    """

    Args:
        model: segmentation model that contains the self-attention MLPs for selecting the features
        to take part in the contrastive learning optimization
        features: Nx256  feature vectors for the contrastive learning (after applying the projection and prediction head)
        class_labels: N corresponding class labels for every feature vector
        num_classes: number of classesin the dataet
        memory: memory bank [List]
        device

    Returns:
        returns the contrastive loss between features vectors from [features] and from [memory] in a class-wise fashion.
    """

    loss = 0

    for c in range(num_classes):
        # get features of an specific class
        mask_c = class_labels == c
        features_c = features[mask_c, :]
        memory_c = memory[c]  # N, 256

        # get the self-attention MLPs both for memory features vectors (projected vectors) and network feature vectors (predicted vectors)
        selector = model.__getattr__('contrastive_class_selector_' + str(c))
        selector_memory = model.__getattr__('contrastive_class_selector_memory' + str(c))

        if memory_c is not None and features_c.shape[0] > 1 and memory_c.shape[0] > 1:
            memory_c = torch.from_numpy(memory_c).to(device)

            # L2 normalize vectors
            memory_c = F.normalize(memory_c, dim=1)  # N, 256
            features_c_norm = F.normalize(features_c, dim=1)  # M, 256

            # compute similarity. All elements with all elements
            similarities = torch.mm(features_c_norm, memory_c.transpose(1, 0))  # MxN
            distances = 1 - similarities  # values between [0, 2] where 0 means same vectors
            # M (elements), N (memory)

            # now weight every sample

            learned_weights_features = selector(features_c.detach())  # detach for trainability
            learned_weights_features_memory = selector_memory(memory_c)

            # self-attention in the memory features-axis and on the learning contrastive features-axis
            learned_weights_features = torch.sigmoid(learned_weights_features)
            rescaled_weights = (learned_weights_features.shape[0] / learned_weights_features.sum(
                dim=0)) * learned_weights_features
            rescaled_weights = rescaled_weights.repeat(1, distances.shape[1])

            distances = distances * rescaled_weights

            learned_weights_features_memory = torch.sigmoid(learned_weights_features_memory)
            learned_weights_features_memory = learned_weights_features_memory.permute(1, 0)
            rescaled_weights_memory = (learned_weights_features_memory.shape[1] / learned_weights_features_memory.sum(
                dim=1)) * learned_weights_features_memory

            rescaled_weights_memory = rescaled_weights_memory.repeat(distances.shape[0], 1)
            distances = distances * rescaled_weights_memory

            loss = loss + distances.mean()

    return loss / num_classes


class FeatureMemory:
    def __init__(self, num_samples, dataset, memory_per_class=2048, feature_size=256, n_classes=19):
        self.num_samples = num_samples
        self.memory_per_class = memory_per_class
        self.feature_size = feature_size
        self.memory = [None] * n_classes
        self.n_classes = n_classes
        if dataset == 'cityscapes':  # usually all classes in one image
            self.per_class_samples_per_image = max(1, int(round(memory_per_class / num_samples)))
        elif dataset == 'pascal_voc':  # usually only around 3 classes on each image, except background class
            self.per_class_samples_per_image = max(1, int(n_classes / 3 * round(memory_per_class / num_samples)))

    def add_features_from_sample_learned(self, model, features, class_labels, batch_size):
        """
        Updates the memory bank with some quality feature vectors per class
        Args:
            model: segmentation model containing the self-attention modules (contrastive_class_selectors)
            features: BxFxWxH feature maps containing the feature vectors for the contrastive (already applied the projection head)
            class_labels:   BxWxH  corresponding labels to the [features]
            batch_size: batch size

        Returns:

        """
        features = features.detach()
        class_labels = class_labels.detach().cpu().numpy()

        elements_per_class = batch_size * self.per_class_samples_per_image

        # for each class, save [elements_per_class]
        for c in range(self.n_classes):
            mask_c = class_labels == c  # get mask for class c
            # selector = model.__getattr__(
            #     'contrastive_class_selector_' + str(c))  # get the self attention module for class c
            features_c = features[mask_c, :]  # get features from class c
            if features_c.shape[0] > 0:
                if features_c.shape[0] > elements_per_class:
                    with torch.no_grad():
                        # get ranking scores
                        # rank = selector(features_c)
                        # rank = torch.sigmoid(rank)
                        # # sort them
                        # _, indices = torch.sort(rank[:, 0], dim=0)
                        # indices = indices.cpu().numpy()
                        # features_c = features_c.cpu().numpy()
                        # # get features with highest rankings
                        # features_c = features_c[indices, :]
                        # new_features = features_c[:elements_per_class, :]

                        # 随机采样
                        features_c = features_c.cpu().numpy()
                        ind = np.arange(len(features_c))
                        sub_ind = np.random.choice(ind, elements_per_class, replace=False)
                        new_features = features_c[sub_ind]

                else:
                    new_features = features_c.cpu().numpy()

                if self.memory[c] is None:  # was empy, first elements
                    self.memory[c] = new_features

                else:  # add elements to already existing list
                    # keep only most recent memory_per_class samples
                    self.memory[c] = np.concatenate((new_features, self.memory[c]), axis=0)[:self.memory_per_class, :]


@UDA.register_module()
class DACL(UDADecorator):

    def __init__(self, **cfg):
        super(DACL, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.ramp_up_iter = cfg['ramp_up_iter']
        self.memory_iter = cfg['memory_iter']
        self.alpha = cfg['alpha']
        self.debug_img_interval = cfg['debug_img_interval']

        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        # Memory Bank
        self.feature_memory = FeatureMemory(num_samples=100, dataset='cityscapes', memory_per_class=256,
                                            feature_size=256,
                                            n_classes=2)
        # 用于计算伪标签损失
        self.unlbl_loss1 = nn.CrossEntropyLoss(reduction="none", ignore_index=255)
        # self.unlbl_loss2 = nn.MSELoss(reduction="none")

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
        log_vars = {}
        all_losses = {}
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

        # 1. 给target_img生成伪标签，使用ema_model
        with torch.no_grad():
            target_result = self.get_ema_model().forward_train(
                target_img, target_img_metas, gt_semantic_seg=None, gt_vec=None, return_last_feat=False,
                return_logits=True,
                dummy_gt_semantic_seg=gt_semantic_seg
            )
            # bs,4,h,w
            target_logits = target_result.pop('decode.logits').detach()

            target_seg_logits = target_logits[:, 2:]

            target_softmax = torch.softmax(target_seg_logits, dim=1)
            target_max_probs, pseudo_labels = torch.max(target_softmax, dim=1)

        # 2. 对target_img进行数据增强，然后使用伪标签训练
        # semiseg进行了两次增强，这里只进行一次
        # class_mix没搞懂，先设置为False
        # 把pseudo_vec和pseudo_seg合并，增强后再分开
        # 4个变量分别 为增强后的图片、对应的伪标签、伪标签置信度、增强的参数(后面没用到)
        target_img_aug, \
        pseudo_label_aug, target_max_prob_aug, target_aug_params, _ = augment_samples(target_img,
                                                                                      pseudo_labels,
                                                                                      probs=target_max_probs,
                                                                                      do_classmix=False,
                                                                                      batch_size=len(
                                                                                          target_img),
                                                                                      ignore_label=255,
                                                                                      weak=False,
                                                                                      )
        # target_img_aug前向计算，使用model，不需要返回损失
        target_result_aug = self.get_model().forward_train(
            target_img_aug, target_img_metas, gt_semantic_seg=None, gt_vec=None, return_last_feat=True,
            return_logits=True,
            dummy_gt_semantic_seg=gt_semantic_seg
        )
        # bs,4,h,w
        target_logits_aug = target_result_aug.pop("decode.logits")

        target_last_feat_aug = target_result_aug.pop("decode.last_feat")

        # 计算伪标签损失
        # 伪标签权重 bs,h,w
        pixel_wise_weight = sigmoid_ramp_up(self.local_iter, self.ramp_up_iter) * torch.ones(
            target_max_prob_aug.shape)
        pixel_wise_weight = pixel_wise_weight.to(dev)
        pixel_wise_weight = (pixel_wise_weight * torch.pow(target_max_prob_aug.detach(), 6))

        loss_pseudo_seg = self.unlbl_loss1(target_logits_aug[:, 2:, ...], pseudo_label_aug)
        loss_pseudo_seg = torch.mean(loss_pseudo_seg * pixel_wise_weight)

        all_losses['loss_pseudo_seg'] = loss_pseudo_seg
        all_losses['pseudo_weight'] = pixel_wise_weight.mean()

        # 3. source_img的有监督训练
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, gt_vec=gt_vec, return_last_feat=True)
        source_last_feat = clean_losses.pop('decode.last_feat')
        all_losses.update(clean_losses)

        # 4. 熵损失，只计算target
        # loss_entropy = entropy_loss(torch.nn.functional.softmax(target_logits_aug, dim=1)) * 0.01
        # all_losses['loss_entropy'] = loss_entropy

        # 5. 对比学习
        # 构建memory bank，使用source图片，ema_model
        if self.local_iter > self.memory_iter:
            with torch.no_grad():
                source_result_ema = self.get_ema_model().forward_train(
                    img, img_metas, gt_semantic_seg=None, gt_vec=None, return_last_feat=True,
                    return_logits=True,
                    dummy_gt_semantic_seg=gt_semantic_seg
                )
                # bs,2,h,w
                source_logits_ema = source_result_ema.pop('decode.logits')[:, 2:].detach()
                source_last_feat_ema = source_result_ema.pop('decode.last_feat')
                source_prob_ema, source_pred_ema = torch.max(
                    torch.softmax(source_logits_ema, dim=1), dim=1
                )

            # 把标签、输出预测、输出概率下采样到last_feat的大小(现在用的网络本来就相等)
            # bs,h,w
            gt_semantic_seg_down = F.interpolate(gt_semantic_seg.float(),
                                                 source_last_feat_ema.shape[-2:],
                                                 mode='nearest'
                                                 ).squeeze(1)
            source_pred_ema_down = F.interpolate(source_pred_ema.float().unsqueeze(1),
                                                 source_last_feat.shape[-2:],
                                                 mode='nearest',
                                                 ).squeeze(1)
            source_prob_ema_down = F.interpolate(source_prob_ema.float().unsqueeze(1),
                                                 source_last_feat.shape[-2:],
                                                 mode='nearest'
                                                 ).squeeze(1)
            # 挑选预测正确并且置信度>0.95
            mask_pred_correctly = (
                    (source_pred_ema_down == gt_semantic_seg_down).float() *
                    (source_prob_ema_down > 0.95).float()
            ).bool()
            source_last_feat_ema_correctly = source_last_feat_ema.permute(0, 2, 3, 1)
            # n,1024
            source_last_feat_ema_correctly = source_last_feat_ema_correctly[mask_pred_correctly, ...]
            # n,
            gt_semantic_seg_down_correctly = gt_semantic_seg_down[mask_pred_correctly]
            # 因为BN，n为1时后续计算会报错
            if source_last_feat_ema_correctly.shape[0] > 1:
                # 计算projected features
                with torch.no_grad():
                    proj = self.get_ema_model().projection_head(source_last_feat_ema_correctly)

                # 更新memory bank
                self.feature_memory.add_features_from_sample_learned(self.get_ema_model(),
                                                                     proj,
                                                                     gt_semantic_seg_down_correctly,
                                                                     len(img)
                                                                     )

        if self.local_iter > self.ramp_up_iter:
            # source图片的对比学习
            mask_valid = gt_semantic_seg_down != 255
            source_last_feat_valid = source_last_feat.permute(0, 2, 3, 1)
            # n,1024
            source_last_feat_valid = source_last_feat_valid[mask_valid, ...]
            # n,
            gt_semantic_seg_down_valid = gt_semantic_seg_down[mask_valid]

            # 计算predicted features
            source_projection = self.get_model().projection_head(source_last_feat_valid)
            source_prediction = self.get_model().prediction_head(source_projection)

            # 计算source对比损失
            loss_contrastive_source = contrastive_class_to_class_learned_memory(self.get_model(),
                                                                                source_prediction,
                                                                                gt_semantic_seg_down_valid,
                                                                                2,
                                                                                self.feature_memory.memory,
                                                                                dev)
            all_losses["loss_contrastive_source"] = loss_contrastive_source * 0.1

            # target图片的对比学习
            pseudo_label_aug_down = F.interpolate(pseudo_label_aug.float().unsqueeze(1),
                                                  target_last_feat_aug.shape[-2:],
                                                  mode='nearest'
                                                  ).squeeze(1)
            valid_mask = pseudo_label_aug_down != 255
            target_last_feat_valid = target_last_feat_aug.permute(0, 2, 3, 1).contiguous()
            # n,1024
            target_last_feat_valid = target_last_feat_valid[valid_mask]
            # n,
            pseudo_label_aug_down_valid = pseudo_label_aug_down[valid_mask]

            # 计算predicted features
            target_projection = self.get_model().projection_head(target_last_feat_valid)
            target_prediction = self.get_model().prediction_head(target_projection)

            # 计算target对比损失
            loss_contrastive_target = contrastive_class_to_class_learned_memory(self.get_model(),
                                                                                target_prediction,
                                                                                pseudo_label_aug_down_valid,
                                                                                2,
                                                                                self.feature_memory.memory,
                                                                                dev
                                                                                )
            all_losses["loss_contrastive_target"] = loss_contrastive_target * 0.1

        all_loss, all_log_vars = self._parse_losses(all_losses)
        log_vars.update(all_log_vars)
        all_loss.backward()

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
