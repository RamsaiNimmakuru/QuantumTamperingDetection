import os, torch, random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ELADataset(Dataset):
    def __init__(self, root, train=True, split=0.8):
        self.samples=[]
        for cls,label in [('authentic',0),('tampered',1)]:
            folder=os.path.join(root,cls)
            imgs=[os.path.join(folder,f) for f in os.listdir(folder) if f.endswith('.png')]
            for p in imgs: self.samples.append((p,label))
        random.shuffle(self.samples)
        cut=int(len(self.samples)*split)
        self.data=self.samples[:cut] if train else self.samples[cut:]
        self.tf=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
    def __len__(self): return len(self.data)
    def __getitem__(self,idx):
        path,label=self.data[idx]
        img=Image.open(path).convert('RGB')
        return self.tf(img),torch.tensor(label,dtype=torch.long)
