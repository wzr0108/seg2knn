import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss
from .contrastive_loss import contrast_preparations


def contrast_criterion(feat,
                       mask,
                       memory,
                       model,
                       use_avg_pool,
                       scale_min_ratio,
                       num_classes,
                       ignore_index):
    # feat Nx512
    # mask N,
    feat, mask = contrast_preparations(feat,
                                       mask,
                                       use_avg_pool,
                                       scale_min_ratio, num_classes,
                                       ignore_index)
    loss = 0
    for c in range(1, num_classes):  # remove background
        # get features of an specific class
        mask_c = mask == c
        feat_c = feat[mask_c, :]  # M, 512
        memory_c = memory[c]  # N, 512

        # get the self-attention MLPs both for memory features vectors (projected vectors) and network feature vectors (predicted vectors)
        # selector = model.__getattr__('contrastive_class_selector_' + str(c))
        # selector_memory = model.__getattr__('contrastive_class_selector_memory' + str(c))
        if memory_c is not None and feat_c.shape[0] > 1 and memory_c.shape[0] > 1:
            memory_c = torch.from_numpy(memory_c).to(feat.device)
            # L2 normalize vectors
            memory_c = F.normalize(memory_c, dim=1)  # N, 512
            features_c_norm = F.normalize(feat_c, dim=1)  # M, 512

            # compute similarity. All elements with all elements
            similarities = torch.mm(features_c_norm, memory_c.transpose(1, 0))  # MxN
            distances = 1 - similarities  # values between [0, 2] where 0 means same vectors

            # weight
            # learned_weights_features = selector(feat_c.detach())  # detach for trainability
            # learned_weights_features_memory = selector_memory(memory_c)

            # self-attention in the memory features-axis and on the learning contrastive features-axis
            # learned_weights_features = torch.sigmoid(learned_weights_features)
            # 和等于M,特征数
            # Mx1
            # rescaled_weights = (learned_weights_features.shape[0] / learned_weights_features.sum(
            #     dim=0)) * learned_weights_features
            # MxN
            # rescaled_weights = rescaled_weights.repeat(1, distances.shape[1])

            # distances = distances * rescaled_weights

            # learned_weights_features_memory = torch.sigmoid(learned_weights_features_memory)
            # 1xN
            # learned_weights_features_memory = learned_weights_features_memory.permute(1, 0)
            # 1xN
            # rescaled_weights_memory = (learned_weights_features_memory.shape[1] / learned_weights_features_memory.sum(
            #     dim=1)) * learned_weights_features_memory
            # MxN
            # rescaled_weights_memory = rescaled_weights_memory.repeat(distances.shape[0], 1)
            # distances = distances * rescaled_weights_memory

            loss += distances.mean()
        else:
            # 避免报错
            loss += 0 * feat.mean()
    return loss


@LOSSES.register_module()
class MemoryBankLoss(nn.Module):
    """ContrastiveLoss.

    Args:

    """

    def __init__(self,
                 use_avg_pool=True,
                 scale_min_ratio=0.75,
                 num_classes=2,
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=255):
        super(MemoryBankLoss, self).__init__()
        assert num_classes is not None
        self.use_avg_pool = use_avg_pool
        self.scale_min_ratio = scale_min_ratio
        self.num_classes = num_classes
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.contrast_criterion = contrast_criterion
        self.ignore_index = ignore_index

    def forward(self,
                auxiliary_head,
                feat,
                mask,
                weight=None,
                ignore_index=255,
                avg_factor=None,
                reduction_override=None,
                memory=None):
        """Forward function."""
        # Parameters mean, covariance are sometimes required
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if isinstance(feat, list):
            loss_contrast = sum(
                self.loss_weight * self.contrast_criterion(
                    feat[i],
                    mask,
                    memory[i],
                    auxiliary_head,
                    use_avg_pool=self.use_avg_pool,
                    scale_min_ratio=self.scale_min_ratio,
                    num_classes=self.num_classes,
                    ignore_index=ignore_index) for i in range(len(feat))
            )
        else:
            loss_contrast = self.loss_weight * self.contrast_criterion(
                feat,
                mask,
                memory,
                auxiliary_head,
                use_avg_pool=self.use_avg_pool,
                scale_min_ratio=self.scale_min_ratio,
                num_classes=self.num_classes,
                ignore_index=ignore_index)
        return loss_contrast
