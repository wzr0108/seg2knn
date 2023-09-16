# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional decode_heads

from .aspp_head import ASPPHead
from .da_head import DAHead
from .daformer_head import DAFormerHead
from .dlv2_head import DLV2Head
from .fcn_head import FCNHead
from .isa_head import ISAHead
from .psp_head import PSPHead
from .segformer_head import SegFormerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .uper_head import UPerHead
from .hv_head import HVHead
from .daformer_head_proto import DAFormerHeadWithProto
from .cellpose_head import CellPoseHead
from .hv2_head import HV2Head
from .segformer_head_pseudo import SegFormerHeadPseudo
from .proj_head import ProjHead

from .discriminator import DomainDiscriminator
from .proj_pred_head import ProjPredHead
from .decoupled_cellpose_head import DecoupledCellPoseHead
from .proj_pred_head_multi import ProjPredHeadMulti
from .cellpose_decouple_head import CellPoseDecoupleHead

__all__ = [
    'FCNHead',
    'PSPHead',
    'ASPPHead',
    'UPerHead',
    'DepthwiseSeparableASPPHead',
    'DAHead',
    'DLV2Head',
    'SegFormerHead',
    'DAFormerHead',
    'ISAHead',
    'HVHead',
    'DomainDiscriminator',
    'DAFormerHeadWithProto',
    'CellPoseHead',
    'HV2Head',
    'SegFormerHeadPseudo',
    'ProjHead',
    'ProjPredHead',
    'DecoupledCellPoseHead',
    'ProjPredHeadMulti',
    'CellPoseDecoupleHead'
]
