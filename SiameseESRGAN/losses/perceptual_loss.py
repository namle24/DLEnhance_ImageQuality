import torch
import torch.nn as nn
from torchvision.models import vgg19


class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights={'conv5_4': 1.0}, use_input_norm=True):
        super().__init__()
        vgg = vgg19(pretrained=True).features.eval()
        self.layers = nn.ModuleDict({
            'conv1_2': vgg[:4],
            'conv2_2': vgg[4:9],
            'conv5_4': vgg[9:35]
        })
        self.layer_weights = layer_weights
        if use_input_norm:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        if hasattr(self, 'mean'):
            pred = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std

        loss = 0
        for name, layer in self.layers.items():
            if name in self.layer_weights:
                pred_feat = layer(pred)
                target_feat = layer(target.detach())
                loss += torch.nn.functional.l1_loss(pred_feat, target_feat) * self.layer_weights[name]
        return loss
