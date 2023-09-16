import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from ..builder import HEADS
from .decode_head_decorator import BaseDecodeHeadDecorator


@HEADS.register_module()
class ProjPredHead(BaseDecodeHeadDecorator):
    """Projection Head for feature dimension reduction in contrastive loss.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 kernel_size=1,
                 dilation=1,
                 **kwargs):
        assert dilation > 0 and isinstance(dilation, int)
        self.kernel_size = kernel_size
        super(ProjPredHead, self).__init__(**kwargs)

        conv_padding = (kernel_size // 2) * dilation
        if self.input_transform == 'multiple_select':
            raise NotImplementedError("multiple_select请使用ProjPredHeadMulti")
            # convs = [[] for _ in range(len(self.in_channels))]
            # for i in range(len(self.in_channels)):
            #     if num_convs > 1:
            #         convs[i].append(
            #             ConvModule(
            #                 self.in_channels[i],
            #                 self.in_channels[i],
            #                 kernel_size=kernel_size,
            #                 padding=conv_padding,
            #                 dilation=dilation,
            #                 conv_cfg=self.conv_cfg,
            #                 norm_cfg=self.norm_cfg,
            #                 act_cfg=self.act_cfg))
            #     convs[i].append(
            #         ConvModule(
            #             self.in_channels[i],
            #             self.channels,
            #             kernel_size=kernel_size,
            #             padding=conv_padding,
            #             dilation=dilation,
            #             conv_cfg=self.conv_cfg,
            #             norm_cfg=self.norm_cfg,
            #             act_cfg=self.act_cfg))
            # if num_convs == 0:
            #     self.convs = nn.ModuleList([nn.Identity() for _ in range(len(self.in_channels))])
            # else:
            #     self.convs = nn.ModuleList([nn.Sequential(*convs[i]) for i in range(len(self.in_channels))])

        else:
            if self.input_transform == 'resize_concat':
                self.mid_channels = self.in_channels // len(self.in_index)
            else:
                self.mid_channels = self.in_channels

            self.projection = nn.Sequential(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation)
            )

            self.prediction = nn.Sequential(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation)
            )

        # 参考文献Semi-Supervised Semantic Segmentation with Pixel-Level Contrastive Learning from a Class-wise Memory Bank
        # 对比学习用的注意力模块, 每类一个Sequential
        for i in range(self.num_classes):
            selector = nn.Sequential(
                nn.Linear(self.channels, self.channels),
                nn.BatchNorm1d(self.channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(self.channels, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(i), selector)

        for i in range(self.num_classes):
            selector = nn.Sequential(
                nn.Linear(self.channels, self.channels),
                nn.BatchNorm1d(self.channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(self.channels, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(i), selector)

    def forward(self, inputs):
        """Forward function."""
        # resize_concat or single_select
        output = self.forward_proj(inputs)
        output = self.forward_pred(output)
        return output

    def forward_proj(self, inputs):
        x = self._transform_inputs(inputs)
        if isinstance(x, list):
            # multiple_select
            raise NotImplementedError("未实现")

        return self.projection(x)

    def forward_pred(self, proj):
        return self.prediction(proj)

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None,
                      **kwargs):
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight, **kwargs)
        return losses

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label, seg_weight=None, **kwargs):
        """Compute segmentation loss."""
        loss = dict()
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        loss['loss_cl'] = self.loss_decode(
            self,
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index,
            **kwargs)
        return loss