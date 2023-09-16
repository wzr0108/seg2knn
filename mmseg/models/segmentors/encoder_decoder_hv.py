# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support for seg_weight
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class EncoderDecoderHV(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 hv_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoderHV, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head, hv_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head, hv_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        #  add a hv head, according to HoverNet
        self.hv_head = builder.build_head(hv_head)

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out_decode_head = self._decode_head_forward_test(x, img_metas)
        out_hv_head = self._hv_head_forward_test(x, img_metas)
        out_decode_head = resize(
            input=out_decode_head,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        out_hv_head = resize(
            input=out_hv_head,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return {"out_decode_head": out_decode_head, "out_hv_head": out_hv_head}

    def _decode_head_forward_train(self,
                                   x,
                                   img_metas,
                                   gt_semantic_seg,
                                   seg_weight=None,
                                   ):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg,
                                                     seg_weight,
                                                     )

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _hv_head_forward_train(self,
                               x,
                               img_metas,
                               gt_semantic_seg,
                               gt_hv_map,
                               seg_weight=None,
                               return_logits=False):
        losses = dict()
        loss_hv = self.hv_head.forward_train(
            x, img_metas, gt_semantic_seg, gt_hv_map,
            self.train_cfg, seg_weight,
        )
        losses.update(add_prefix(loss_hv, 'hv'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _hv_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for hv head in
        inference."""
        seg_logits = self.hv_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self,
                                      x,
                                      img_metas,
                                      gt_semantic_seg,
                                      seg_weight=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg, seg_weight)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg=None,
                      seg_weight=None,
                      gt_hv_map=None,
                      return_feat=False,):
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
        x = self.extract_feat(img)

        losses = dict()
        if return_feat:
            losses['features'] = x

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg,
                                                      seg_weight)
        losses.update(loss_decode)
        if gt_hv_map is not None:
            loss_hv = self._hv_head_forward_train(x, img_metas, gt_semantic_seg, gt_hv_map, seg_weight)
            losses.update(loss_hv)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, seg_weight)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        # preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        preds = {
            'out_decode_head': img.new_zeros((batch_size, num_classes, h_img, w_img)),
            'out_hv_head': img.new_zeros((batch_size, 2, h_img, w_img))
        }
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)

                for k, v in crop_seg_logit.items():
                    preds[k] += F.pad(v,
                                      (int(x1), int(preds[k].shape[3] - x2), int(y1),
                                       int(preds[k].shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        for k, v in crop_seg_logit.items():
            preds[k] = preds[k] / count_mat
        if rescale:
            for k, v in preds.items():
                preds[k] = resize(
                    v,
                    size=img_meta[0]['ori_shape'][:2],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            if isinstance(seg_logit, dict):
                for k, v in seg_logit.items():
                    seg_logit[k] = resize(
                        v,
                        size=size,
                        mode='bilinear',
                        align_corners=self.align_corners,
                        warning=False)
            else:
                seg_logit = resize(
                    seg_logit,
                    size=size,
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit['out_decode_head'], dim=1)
        hv_map = seg_logit['out_hv_head']
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
                hv_map = hv_map.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))
                hv_map = hv_map.flip(dims=(2,))

        return output, hv_map

    def simple_test(self, img, img_meta, rescale=True, save_results=True):
        """Simple test with single image."""
        seg_logit, hv_map = self.inference(img, img_meta, rescale)
        # ---------------------------------------------------------------#
        # seg_logit是softmax的结果
        # 保存seg_logit 和 hv_map
        if save_results:
            seg_prob = seg_logit.permute(0, 2, 3, 1).contiguous()
            hv_map = hv_map.permute(0, 2, 3, 1).contiguous()
            seg_prob = seg_prob[0].cpu().numpy()
            hv_map = hv_map[0].cpu().numpy()
            hv1 = hv_map[..., 0]
            hv2 = hv_map[..., 1]
            img_filename = img_meta[0]["filename"]
            save_dir = os.path.join(os.path.dirname(os.path.dirname(img_filename)), 'npy_results')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, os.path.basename(img_filename).replace("png", "npy"))
            # 只保存前景概率
            save_file = np.concatenate([seg_prob[..., 1:], hv_map], -1)
            np.save(save_path, save_file)

            hv_map = torch.from_numpy(hv_map).unsqueeze(0).to(seg_logit.device)
            hv_map = hv_map.permute(0, 3, 1, 2).contiguous()
        # ---------------------------------------------------------------#

        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred

        # 拼接seg 和 hv
        seg_pred = torch.cat([seg_logit, hv_map], 1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)

        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
