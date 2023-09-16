import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class EncoderDecoderTamper(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 feat_dim=256,
                 with_cls=False
                 ):
        super(EncoderDecoderTamper, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head
        # 参考文献Semi-Supervised Semantic Segmentation with Pixel-Level Contrastive Learning from a Class-wise Memory Bank
        # 对比学习用的注意力模块, 每类一个Sequential
        # dim_in = decode_head['decoder_params']["embed_dim"]
        # for i in range(2):
        #     selector = nn.Sequential(
        #         nn.Linear(feat_dim, feat_dim),
        #         nn.BatchNorm1d(feat_dim),
        #         nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #         nn.Linear(feat_dim, 1)
        #     )
        #     self.__setattr__('contrastive_class_selector_' + str(i), selector)
        #
        # for i in range(2):
        #     selector = nn.Sequential(
        #         nn.Linear(feat_dim, feat_dim),
        #         nn.BatchNorm1d(feat_dim),
        #         nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #         nn.Linear(feat_dim, 1)
        #     )
        #     self.__setattr__('contrastive_class_selector_memory' + str(i), selector)
        #
        # self.projection_head = nn.Sequential(
        #     nn.Linear(dim_in, feat_dim),
        #     nn.BatchNorm1d(feat_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(feat_dim, feat_dim)
        # )
        # self.prediction_head = nn.Sequential(
        #     nn.Linear(feat_dim, feat_dim),
        #     nn.BatchNorm1d(feat_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(feat_dim, feat_dim)
        # )

        self.with_cls = with_cls

        if self.with_cls:
            self.cls = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
                nn.Linear(in_features=256, out_features=1, bias=True),
            )
            self.cls_loss = nn.BCEWithLogitsLoss()

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

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
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self,
                                   x,
                                   img_metas,
                                   gt_semantic_seg,
                                   seg_weight=None,
                                   return_last_feat=False,
                                   return_logits=False,
                                   img_shape=None
                                   ):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg,
                                                     seg_weight,
                                                     return_last_feat=return_last_feat,
                                                     return_logits=return_logits,
                                                     img_shape=img_shape)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
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
                      gt_semantic_seg,
                      seg_weight=None,
                      return_feat=False,
                      return_last_feat=False,
                      return_logits=False,
                      img_shape=None,
                      ):
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
        xs = self.extract_feat(img)

        if not isinstance(xs[0], (list, tuple)):
            xs = [xs, ]
            loss_weight = None
        else:
            loss_weight = [0.5, 1]

        def upd_loss(losses, idx, weight):
            new_losses = dict()
            for k, v in losses.items():
                new_k = '{}{}'.format(k, idx)
                if weight != 1 and 'loss' in k:
                    new_k = '{}_w{}'.format(new_k, weight)
                if isinstance(v, list) or isinstance(v, tuple):
                    if "loss" in new_k:
                        new_losses[new_k] = [i * weight for i in v]
                    else:
                        new_losses[new_k] = v
                else:
                    if "loss" in new_k:
                        new_losses[new_k] = v * weight
                    else:
                        new_losses[new_k] = v
            return new_losses

        losses = dict()
        for i, x in enumerate(xs):
            if return_feat:
                losses['features'] = x

            # if self.with_cls:
            #     print(gt_cls)
            #     for meta in img_metas:
            #         print(meta["ori_filename"])
            #     print("\n")
            #     cls_output = self.cls(x[-1])
            #     loss_cls = self.cls_loss(cls_output, gt_cls)
            #     losses["cls.loss_bce"] = loss_cls

            loss_decode = self._decode_head_forward_train(x, img_metas,
                                                          gt_semantic_seg,
                                                          seg_weight, return_logits=return_logits,
                                                          return_last_feat=return_last_feat,
                                                          img_shape=img_shape)
            if len(xs) > 1:
                loss_decode = upd_loss(loss_decode, idx=i, weight=loss_weight[i])
            losses.update(loss_decode)

            if self.with_auxiliary_head:
                loss_aux = self._auxiliary_head_forward_train(
                    x, img_metas, gt_semantic_seg, seg_weight)
                if len(xs) > 1:
                    loss_aux = upd_loss(loss_aux, idx=i, weight=loss_weight[i])
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
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
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
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
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
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)

        seg_logit = seg_logit.cpu().numpy()
        # unravel batch dim
        seg_logit = list(seg_logit)
        return seg_logit

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

        seg_logit = seg_logit.cpu().numpy()
        # unravel batch dim
        seg_logit = list(seg_logit)
        return seg_logit

    # def train(self, mode=True):
    #     """Convert the model into training mode while keep layers freezed."""
    #     super(EncoderDecoderTamper, self).train(mode)
    #     for m in self.modules():
    #         # trick: eval have effect on BatchNorm only
    #         if isinstance(m, _BatchNorm):
    #             m.eval()
