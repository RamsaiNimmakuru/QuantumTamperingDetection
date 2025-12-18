import os, torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class RestorationDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir):
        self.deg=[os.path.join(degraded_dir,f) for f in os.listdir(degraded_dir)]
        self.cln=[os.path.join(clean_dir,f) for f in os.listdir(clean_dir)]
        self.tf=transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])
    def __len__(self): return min(len(self.deg), len(self.cln))
    def __getitem__(self,idx):
        d=Image.open(self.deg[idx]).convert('RGB')
        c=Image.open(self.cln[idx]).convert('RGB')
        return self.tf(d), self.tf(c)
