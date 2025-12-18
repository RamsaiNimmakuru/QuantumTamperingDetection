import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        def CBR(in_c,out_c):
            return nn.Sequential(nn.Conv2d(in_c,out_c,3,1,1),
                                 nn.BatchNorm2d(out_c),nn.ReLU(True))
        self.enc1=CBR(3,64); self.enc2=CBR(64,128)
        self.enc3=CBR(128,256); self.enc4=CBR(256,512)
        self.pool=nn.MaxPool2d(2)
        self.up3=nn.ConvTranspose2d(512,256,2,2)
        self.up2=nn.ConvTranspose2d(256,128,2,2)
        self.up1=nn.ConvTranspose2d(128,64,2,2)
        self.dec3=CBR(512,256); self.dec2=CBR(256,128); self.dec1=CBR(128,64)
        self.out=nn.Conv2d(64,3,1)
    def forward(self,x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1))
        e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
        d3=self.dec3(torch.cat([self.up3(e4),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return torch.sigmoid(self.out(d1))
