from .l1_loss import L1LossMod
from .reg_loss import RegLoss
from .tv_loss import TVLoss
from .ddpm_loss import DDPMMSELossMod

from .perceptual_loss import PerceptualLoss, perceptual_loss_fn
from .normal_loss import NormalLoss, normal_consistency_loss, normal_smoothness_loss
from .edge_loss import EdgeLoss
__all__ = [
    'L1LossMod', 'RegLoss', 'DDPMMSELossMod', 'TVLoss',
    'NormalLoss', 'PerceptualLoss', 'perceptual_loss_fn',
    'normal_consistency_loss', 'normal_smoothness_loss','EdgeLoss'
]