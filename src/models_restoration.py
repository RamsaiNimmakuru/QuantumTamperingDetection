# src/models_restoration.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn_relu(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
        )
    def forward(self, x):
        return F.relu(x + self.net(x))

class ResUNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 7, padding=3), nn.ReLU(inplace=True))
        self.enc2 = conv_bn_relu(base, base*2, 3, 2, 1)
        self.enc3 = conv_bn_relu(base*2, base*4, 3, 2, 1)
        self.ress = nn.Sequential(*[ResBlock(base*4) for _ in range(4)])
        self.up3 = nn.ConvTranspose2d(base*4, base*2, 4, stride=2, padding=1)
        self.dec3 = conv_bn_relu(base*4, base*2)
        self.up2 = nn.ConvTranspose2d(base*2, base, 4, 2, 1)
        self.dec2 = conv_bn_relu(base*2, base)
        self.outc = nn.Conv2d(base, out_ch, 7, padding=3)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        c = self.ress(e3)
        d3 = F.relu(self.dec3(torch.cat([self.up3(c), e2], dim=1)))
        d2 = F.relu(self.dec2(torch.cat([self.up2(d3), e1], dim=1)))
        return torch.tanh(self.outc(d2))

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=6, base=64, n_layers=3):
        super().__init__()
        kw=4; pad=1
        layers = [nn.Conv2d(in_ch, base, kw, stride=2, padding=pad), nn.LeakyReLU(0.2, True)]
        nf = base
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf*2, 512)
            stride = 2 if n < n_layers-1 else 1
            layers += [nn.Conv2d(nf_prev, nf, kw, stride=stride, padding=pad),
                       nn.BatchNorm2d(nf), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(nf, 1, kw, stride=1, padding=pad)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
