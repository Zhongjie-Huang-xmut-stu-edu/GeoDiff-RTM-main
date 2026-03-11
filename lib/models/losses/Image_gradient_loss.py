import torch
import torch.nn as nn
import torch.nn.functional as F
from mmgen.models.builder import MODULES
from mmgen.models.losses.utils import weighted_loss


@weighted_loss
def normal_consistency_loss(normals1, normals2):
    """
    计算两组法向量的一致性损失
    Args:
        normals1, normals2: [N, 3] 法向量
    """
    # 计算法向量之间的余弦相似度
    cosine_sim = torch.sum(normals1 * normals2, dim=-1)
    # 1 - cosine_similarity，越相似损失越小
    loss = 1.0 - cosine_sim.abs()
    return loss.mean()


@weighted_loss  
def normal_smoothness_loss(normals, coords):
    """
    计算法向量的平滑性损失
    Args:
        normals: [N, 3] 法向量
        coords: [N, 3] 对应的3D坐标
    """
    if normals.size(0) < 2:
        return torch.tensor(0.0, device=normals.device)
    
    # 计算相邻点的法向量差异
    normal_diff = normals[1:] - normals[:-1]
    coord_diff = coords[1:] - coords[:-1]
    
    # 距离加权的平滑性损失
    distances = torch.norm(coord_diff, dim=-1, keepdim=True)
    weights = torch.exp(-distances)  # 距离越近权重越大
    
    smoothness = torch.norm(normal_diff, dim=-1) * weights.squeeze()
    return smoothness.mean()


def compute_image_gradients(images):
    """
    计算图像梯度作为法向量的代理
    Args:
        images: [B, C, H, W] 图像
    Returns:
        gradients: [B, 2, H, W] x和y方向的梯度
    """
    # Sobel算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
    
    # 转换为灰度图
    if images.size(1) == 3:
        gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
    else:
        gray = images
    
    # 计算梯度
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    
    return torch.cat([grad_x, grad_y], dim=1)


@MODULES.register_module()
class NormalLoss(nn.Module):
    """法向量损失"""
    
    def __init__(self, 
                 loss_type='consistency',  # 'consistency' or 'smoothness'
                 loss_weight=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.loss_weight = loss_weight
    
    def forward(self, pred, target, **kwargs):
        """
        计算法向量损失
        Args:
            pred: 预测图像 [B, C, H, W] 或者法向量 [N, 3]
            target: 目标图像 [B, C, H, W] 或者法向量 [N, 3]
        """
        print(f"NormalLoss: pred shape: {pred.shape}, dtype: {pred.dtype}, min: {pred.min()}, max: {pred.max()}")
        print(f"NormalLoss: target shape: {target.shape}, dtype: {target.dtype}, min: {target.min()}, max: {target.max()}")
        if pred.dim() == 4:  # 图像输入
            # 使用图像梯度作为法向量的代理
            pred_gradients = compute_image_gradients(pred)
            target_gradients = compute_image_gradients(target)
            
            if self.loss_type == 'consistency':
                # 梯度一致性损失
                loss = F.mse_loss(pred_gradients, target_gradients)
            elif self.loss_type == 'smoothness':
                # 梯度平滑性损失
                # 计算梯度的变化
                pred_grad_diff_x = pred_gradients[:, :, 1:] - pred_gradients[:, :, :-1]
                pred_grad_diff_y = pred_gradients[:, :, :, 1:] - pred_gradients[:, :, :, :-1]
                loss = pred_grad_diff_x.abs().mean() + pred_grad_diff_y.abs().mean()
            else:
                loss = torch.tensor(0.0, device=pred.device)
                
        elif pred.dim() == 2 and pred.size(-1) == 3:  # 法向量输入
            if self.loss_type == 'consistency':
                loss = normal_consistency_loss(pred, target)
            elif self.loss_type == 'smoothness':
                # 需要额外的坐标信息
                coords = kwargs.get('coords', None)
                if coords is not None:
                    loss = normal_smoothness_loss(pred, coords)
                else:
                    loss = torch.tensor(0.0, device=pred.device)
            else:
                loss = torch.tensor(0.0, device=pred.device)
        else:
            loss = torch.tensor(0.0, device=pred.device)
        
        return loss * self.loss_weight