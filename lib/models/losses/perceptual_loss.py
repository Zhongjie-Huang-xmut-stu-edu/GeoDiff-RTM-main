import torch
import torch.nn as nn
import torchvision.models as models
from mmgen.models.builder import MODULES
from mmgen.models.losses.utils import weighted_loss


@weighted_loss
def perceptual_loss_fn(pred, target, feature_extractor):
    
    loss = 0.0
    
    
    pred_features = feature_extractor(pred)
    target_features = feature_extractor(target)
    
    
    for pred_feat, target_feat in zip(pred_features, target_features):
        loss += torch.nn.functional.mse_loss(pred_feat, target_feat)
    
    return loss


@MODULES.register_module()
class PerceptualLoss(nn.Module):
    
    
    def __init__(self, 
                 feature_layers=['conv2_2', 'conv3_3', 'conv4_3'],
                 loss_weight=1.0,
                 pretrained=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.feature_layers = feature_layers
        
       
        vgg = models.vgg16(pretrained=pretrained).features
        
        
        self.feature_extractor = nn.ModuleDict()
        
        
        layer_map = {
            'conv1_1': 0, 'conv1_2': 2,
            'conv2_1': 5, 'conv2_2': 7,
            'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14,
            'conv4_1': 17, 'conv4_2': 19, 'conv4_3': 21,
            'conv5_1': 24, 'conv5_2': 26, 'conv5_3': 28,
        }
        
        
        for layer_name in feature_layers:
            if layer_name in layer_map:
                end_layer = layer_map[layer_name]
                self.feature_extractor[layer_name] = nn.Sequential(
                    *list(vgg.children())[:end_layer + 1]
                )
        
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def extract_features(self, x):
        
        features = []
        for layer_name in self.feature_layers:
            if layer_name in self.feature_extractor:
                feat = self.feature_extractor[layer_name](x)
                features.append(feat)
        return features
    
    def forward(self, pred, target, **kwargs):
       
        
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        
        loss = 0.0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += torch.nn.functional.mse_loss(pred_feat, target_feat)
        
        return loss * self.loss_weight