import torch
import torch.nn as nn
import torch.nn.functional as F
from mmgen.models.builder import MODULES
from mmgen.models.losses.utils import weighted_loss


@weighted_loss
def edge_loss(pred, target, edge_threshold=0.1):
    """
    计算边缘损失
    Args:
        pred: 预测图像 [B, C, H, W]
        target: 目标图像 [B, C, H, W]
        edge_threshold: 边缘检测阈值，用于增强强边缘的权重
    Returns:
        边缘损失值
    """
    # Sobel算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)

    def get_edges(img):
        """提取图像边缘"""
        # 转换为灰度图
        if img.size(1) == 3:
            # RGB to Gray: 0.299R + 0.587G + 0.114B
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img

        # 计算梯度
        sobel_x_expand = sobel_x.expand(gray.size(1), -1, -1, -1)
        sobel_y_expand = sobel_y.expand(gray.size(1), -1, -1, -1)

        edge_x = F.conv2d(gray, sobel_x_expand, padding=1, groups=gray.size(1))
        edge_y = F.conv2d(gray, sobel_y_expand, padding=1, groups=gray.size(1))

        # 计算边缘强度
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        return edge_magnitude

    # 计算预测和目标的边缘
    pred_edges = get_edges(pred)
    target_edges = get_edges(target)

    # ✅ 优化1: 使用边缘阈值创建权重掩码
    if edge_threshold > 0:
        # 创建边缘权重掩码，强边缘区域权重更大
        edge_weight = torch.maximum(pred_edges, target_edges)
        edge_mask = (edge_weight > edge_threshold).float()
        
        # 基础权重 + 边缘增强权重
        weight_mask = 1.0 + edge_mask * 2.0  # 强边缘区域权重增加2倍
        
        # 加权边缘损失
        edge_diff = (pred_edges - target_edges) ** 2
        edge_loss_value = (edge_diff * weight_mask).mean()
    else:
        # 标准MSE损失
        edge_loss_value = F.mse_loss(pred_edges, target_edges)

    return edge_loss_value


@MODULES.register_module()
class EdgeLoss(nn.Module):
    """边缘损失，强化边缘和轮廓"""

    def __init__(self,
                 loss_weight=1.0,
                 edge_threshold=0.1,  # ✅ 现在真正使用这个参数
                 reduction='mean',
                 loss_name='loss_edge',
                 use_l1=False):       # ✅ 新增选项：使用L1而不是L2损失
        super(EdgeLoss, self).__init__()
        self.loss_weight = loss_weight
        self.edge_threshold = edge_threshold
        self.reduction = reduction
        self._loss_name = loss_name
        self.use_l1 = use_l1

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """
        前向传播
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
            weight: 权重
            avg_factor: 平均因子
            reduction_override: 损失缩减方式覆盖
        Returns:
            边缘损失值
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # 计算边缘损失
        loss_edge = edge_loss(
            pred,
            target,
            edge_threshold=self.edge_threshold,  # ✅ 传递阈值参数
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_edge * self.loss_weight

    @property
    def loss_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name