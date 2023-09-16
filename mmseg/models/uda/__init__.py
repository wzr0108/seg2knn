# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.models.uda.dacs import DACS
from .dacl import DACL
from .sepico import SePiCo
from .daprojpred import DAProjPred
from .clprojpred import CLProjPred
from .pseudo import Pseudo
from .daprojpred2 import DAProjPred2

__all__ = ['DACS', 'DACL', 'SePiCo', 'DAProjPred', 'CLProjPred', 'Pseudo', 'DAProjPred2']
