# scripts/finetune_columbia.py
"""
Robust finetune script for Columbia ELA dataset.
Detects classifier head location automatically and unfreezes it.
Saves best model to models/vgg16_columbia_finetuned.pt
"""
import os, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Adjust these if needed
ELA_BASE = "data/processed/ELA_columbia"
MODEL_IN = "models/vgg16_baseline.pt"
MODEL_OUT = "models/vgg16_columbia_finetuned.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 16
LR = 5e-5
EPOCHS = 4

# Import model builder (adjust if different)
from src.model_vgg import build_vgg16

# transform (ImageNet normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class ELADataset(Dataset):
    def __init__(self, auth_dir, tamp_dir, tf=transform):
        self.items=[]
        for f in sorted(Path(auth_dir).iterdir()):
            if f.suffix.lower() in (".jpg",".jpeg",".png"):
                self.items.append((str(f), 0))
        for f in sorted(Path(tamp_dir).iterdir()):
            if f.suffix.lower() in (".jpg",".jpeg",".png"):
                self.items.append((str(f), 1))
        self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p, l = self.items[idx]
        img = Image.open(p).convert("RGB")
        return self.tf(img), torch.tensor(l, dtype=torch.long)

def find_and_unfreeze_classifier(model):
    """
    Find classifier layers and unfreeze them.
    Returns True if found and unfrozen, else False.
    """
    # common attribute names to try
    candidates = [
        "classifier",     # plain vgg
        "model.classifier", # wrapper storing vgg under .model
        "model.classifier", # duplicate safe
        "net.classifier",
        "module.classifier",
        "features.classifier",
    ]
    # also try for nested objects: object.model.classifier etc.
    def getattr_chain(obj, path):
        cur = obj
        for part in path.split('.'):
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur

    for cand in candidates:
        head = getattr_chain(model, cand)
        if head is not None:
            # set requires_grad = True for parameters in head
            for p in head.parameters():
                p.requires_grad = True
            print(f"Unfroze classifier at: {cand}")
            return True

    # fallback: try common attribute names without dot
    for name in ["classifier", "fc", "head", "projection", "proj"]:
        if hasattr(model, name):
            head = getattr(model, name)
            for p in head.parameters():
                p.requires_grad = True
            print(f"Unfroze classifier at attribute: {name}")
            return True

    # fallback: search for any nn.Sequential module with Linear layers near the end (heuristic)
    import torch.nn as nn
    for attr in dir(model):
        try:
            obj = getattr(model, attr)
            if isinstance(obj, nn.Sequential):
                # check if last layers contain Linear
                for sub in list(obj)[-3:]:
                    if isinstance(sub, nn.Linear):
                        for p in obj.parameters():
                            p.requires_grad = True
                        print(f"Unfroze Sequential classifier found at attribute: {attr}")
                        return True
        except Exception:
            continue

    # final fallback: unfreeze all params (not ideal)
    print("WARNING: Could not find classifier head automatically. As fallback will unfreeze all parameters.")
    for p in model.parameters():
        p.requires_grad = True
    return False

def smart_load_state_dict(model, ckpt_path, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        # ckpt could be full model object
        try:
            model.load_state_dict(ckpt.state_dict())
            return model
        except Exception:
            return ckpt
    try:
        model.load_state_dict(ckpt)
        print("Loaded checkpoint directly.")
        return model
    except Exception:
        pass
    # try prefixing 'model.'
    prefixed = {"model."+k:v for k,v in ckpt.items()}
    try:
        model.load_state_dict(prefixed, strict=False)
        print("Loaded checkpoint by prefixing 'model.'")
        return model
    except Exception:
        pass
    # try stripping 'model.'
    stripped = {}
    for k,v in ckpt.items():
        if k.startswith("model."):
            stripped[k[len("model."):]] = v
        else:
            stripped[k] = v
    try:
        model.load_state_dict(stripped, strict=False)
        print("Loaded checkpoint by stripping 'model.'")
        return model
    except Exception:
        pass
    # fallback
    model.load_state_dict(ckpt, strict=False)
    print("Loaded checkpoint with strict=False fallback.")
    return model

def main():
    # check data paths
    train_auth = os.path.join(ELA_BASE, "authentic", "train")
    train_tamp = os.path.join(ELA_BASE, "tampered", "train")
    val_auth = os.path.join(ELA_BASE, "authentic", "val")
    val_tamp = os.path.join(ELA_BASE, "tampered", "val")
    assert os.path.isdir(train_auth) and os.path.isdir(train_tamp), "Train folders missing"
    assert os.path.isdir(val_auth) and os.path.isdir(val_tamp), "Val folders missing"

    train_ds = ELADataset(train_auth, train_tamp)
    val_ds = ELADataset(val_auth, val_tamp)
    tr = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    vl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2)

    model = build_vgg16(num_classes=2)
    model = smart_load_state_dict(model, MODEL_IN, map_location=DEVICE)
    model.to(DEVICE)

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze classifier intelligently
    find_and_unfreeze_classifier(model)

    # Build optimizer only for params with requires_grad=True
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameter count: {sum(p.numel() for p in trainable)}")
    optimizer = torch.optim.Adam(trainable, lr=LR, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0; total_samples = 0
        for x,y in tqdm(tr, desc=f"Train Epoch {epoch+1}"):
            x = x.to(DEVICE); y = y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            total_loss += loss.item()*x.size(0); total_samples += x.size(0)
        avg_loss = total_loss/total_samples if total_samples>0 else 0.0
        print(f"Epoch {epoch+1} train_loss {avg_loss:.4f}")

        # Validation
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x,y in vl:
                x = x.to(DEVICE); y = y.to(DEVICE)
                pred = torch.argmax(model(x), dim=1)
                correct += (pred==y).sum().item(); total += y.size(0)
        acc = correct/total if total>0 else 0.0
        print(f"Epoch {epoch+1} val_acc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_OUT)
            print("Saved best model ->", MODEL_OUT)

    print("Done. Best val acc:", best_acc)

if __name__ == "__main__":
    main()
