import torch
import torch.nn as nn
import torch.nn.functional as F
from mmgen.models.builder import MODULES
from mmgen.models.losses.utils import weighted_loss


@weighted_loss
def edge_loss(pred, target, edge_threshold=0.1):
    
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)

    def get_edges(img):
        
        if img.size(1) == 3:
            # RGB to Gray: 0.299R + 0.587G + 0.114B
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img

        
        sobel_x_expand = sobel_x.expand(gray.size(1), -1, -1, -1)
        sobel_y_expand = sobel_y.expand(gray.size(1), -1, -1, -1)

        edge_x = F.conv2d(gray, sobel_x_expand, padding=1, groups=gray.size(1))
        edge_y = F.conv2d(gray, sobel_y_expand, padding=1, groups=gray.size(1))

        
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        return edge_magnitude

    
    pred_edges = get_edges(pred)
    target_edges = get_edges(target)

    
    if edge_threshold > 0:
        
        edge_weight = torch.maximum(pred_edges, target_edges)
        edge_mask = (edge_weight > edge_threshold).float()
        
        
        weight_mask = 1.0 + edge_mask * 2.0  
        
        
        edge_diff = (pred_edges - target_edges) ** 2
        edge_loss_value = (edge_diff * weight_mask).mean()
    else:
        
        edge_loss_value = F.mse_loss(pred_edges, target_edges)

    return edge_loss_value


@MODULES.register_module()
class EdgeLoss(nn.Module):
    
    def __init__(self,
                 loss_weight=1.0,
                 edge_threshold=0.1,  
                 reduction='mean',
                 loss_name='loss_edge',
                 use_l1=False):       
        super(EdgeLoss, self).__init__()
        self.loss_weight = loss_weight
        self.edge_threshold = edge_threshold
        self.reduction = reduction
        self._loss_name = loss_name
        self.use_l1 = use_l1

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        
        loss_edge = edge_loss(
            pred,
            target,
            edge_threshold=self.edge_threshold,  
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