# scripts/extract_classical_features.py
import os, numpy as np, torch
from torchvision import transforms
from PIL import Image
from src.model_quantum import HybridQuantumCNN
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HybridQuantumCNN().to(device)
# load pretrained backbone if you want to use frozen feature extractor
# model.load_state_dict(torch.load("models/quantum_cnn.pt", map_location=device))

# feature extractor is model.feature_extractor + model.pool
fe = model.feature_extractor.eval()
pool = model.pool

tf = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])

def extract(folder):
    feats=[]; labels=[]
    for cls, lab in [("authentic",0), ("tampered",1)]:
        path = os.path.join(folder, cls)
        for f in os.listdir(path):
            if not f.lower().endswith((".jpg",".png",".tif")): continue
            img = Image.open(os.path.join(path,f)).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = fe(x)
                out = pool(out).squeeze(-1).squeeze(-1).cpu().numpy().ravel()
            feats.append(out)
            labels.append(lab)
    return np.array(feats), np.array(labels)

os.makedirs("data/features", exist_ok=True)
train_feats, train_labels = extract("data/processed/ELA_finetune/train")
val_feats, val_labels = extract("data/processed/ELA_finetune/val")
np.save("data/features/train_feats.npy", train_feats)
np.save("data/features/train_labels.npy", train_labels)
np.save("data/features/val_feats.npy", val_feats)
np.save("data/features/val_labels.npy", val_labels)
print("Saved classical features shapes:", train_feats.shape, val_feats.shape)
