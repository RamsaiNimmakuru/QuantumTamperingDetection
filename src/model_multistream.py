# src/model_multistream.py
import torch
import torch.nn as nn
import torchvision.models as models

def adapt_first_conv(vgg_model, new_in_channels):
    old = vgg_model.features[0]
    if old.in_channels == new_in_channels:
        return vgg_model
    new_conv = nn.Conv2d(new_in_channels, old.out_channels,
                         kernel_size=old.kernel_size, stride=old.stride,
                         padding=old.padding, bias=(old.bias is not None))
    with torch.no_grad():
        old_w = old.weight.data
        if new_in_channels > old.in_channels:
            new_w = torch.zeros((old.out_channels, new_in_channels, old_w.shape[2], old_w.shape[3]))
            new_w[:, :old.in_channels, :, :] = old_w
            mean_ch = old_w.mean(dim=1, keepdim=True)
            for c in range(old.in_channels, new_in_channels):
                new_w[:, c:c+1, :, :] = mean_ch
            new_conv.weight.data = new_w
        else:
            new_conv.weight.data = old_w[:, :new_in_channels, :, :].clone()
        if old.bias is not None:
            new_conv.bias.data = old.bias.data.clone()
    vgg_model.features[0] = new_conv
    return vgg_model

class StreamBackbone(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()
        vgg = models.vgg16(pretrained=pretrained)
        vgg = adapt_first_conv(vgg, in_channels)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.out_features = 25088

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class SmallCNN(nn.Module):
    def __init__(self, in_channels=3, out_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.out_dim = 128
        self.fc = nn.Linear(self.out_dim, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MultiStreamFusion(nn.Module):
    def __init__(self, pretrained_rgb=True, feat_dim=25088, hidden=1024, n_classes=2):
        super().__init__()
        self.rgb = StreamBackbone(in_channels=3, pretrained=pretrained_rgb)
        self.ela = SmallCNN(in_channels=3, out_dim=1024)
        self.res = SmallCNN(in_channels=3, out_dim=1024)
        total = self.rgb.out_features + 1024 + 1024
        self.fc = nn.Sequential(
            nn.Linear(total, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, rgb, ela, residual):
        fr = self.rgb(rgb)
        fe = self.ela(ela)
        frs = self.res(residual)
        fused = torch.cat([fr, fe, frs], dim=1)
        out = self.fc(fused)
        return out
