'''
import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights

def build_vgg16(num_classes=2):
    model=vgg16(weights=VGG16_Weights.DEFAULT)
    for p in model.features.parameters():
        p.requires_grad=False
    model.classifier[-1]=nn.Linear(4096,num_classes)
    return model
    
'''
import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights

class TamperClassifierVGG16(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = vgg16(weights=VGG16_Weights.DEFAULT)

        for p in base.features.parameters():
            p.requires_grad = False     # freeze backbone

        # Replace classifier output
        base.classifier[-1] = nn.Linear(4096, num_classes)

        self.model = base
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

def build_vgg16(num_classes=2):
    return TamperClassifierVGG16(num_classes)
