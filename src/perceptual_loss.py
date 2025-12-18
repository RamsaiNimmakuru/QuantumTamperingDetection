# src/perceptual_loss.py
import torch
import torch.nn as nn
from torchvision import models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device='cpu', layer_indices=(3,8,17)):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.layer_indices = set(layer_indices)
        self.device = device

    def forward(self, x, y):
        # x,y in [-1,1] -> convert to [0,1]
        x = (x + 1) * 0.5
        y = (y + 1) * 0.5
        feats_x = []
        feats_y = []
        outx = x; outy = y
        for i, layer in enumerate(self.vgg):
            outx = layer(outx); outy = layer(outy)
            if i in self.layer_indices:
                feats_x.append(outx); feats_y.append(outy)
        loss = 0.0
        for fx, fy in zip(feats_x, feats_y):
            loss += nn.functional.l1_loss(fx, fy)
        return loss
