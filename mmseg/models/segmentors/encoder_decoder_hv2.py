from ..builder import SEGMENTORS
from .encoder_decoder_cellpose import EncoderDecoderCellPose


@SEGMENTORS.register_module()
class EncoderDecoderHV2(EncoderDecoderCellPose):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 ):
        super(EncoderDecoderHV2, self).__init__(
            backbone,
            decode_head,
            neck,
            auxiliary_head,
            train_cfg,
            test_cfg,
            pretrained,
            init_cfg
        )

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg=None,
                      seg_weight=None,
                      gt_hv_map=None,
                      return_feat=False,
                      mode="dec",
                      **kwargs):
        assert mode in ["all", "aux", "dec"]
        x = self.extract_feat(img)

        losses = dict()
        if return_feat:
            losses['features'] = x
        if mode == "all" or mode == "dec":
            loss_decode = self._decode_head_forward_train(x, img_metas,
                                                          gt_semantic_seg, gt_hv_map,
                                                          seg_weight,
                                                          )
            losses.update(loss_decode)

        if self.with_auxiliary_head and (mode == "all" or mode == "aux"):
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, seg_weight, **kwargs)
            losses.update(loss_aux)

        return losses
