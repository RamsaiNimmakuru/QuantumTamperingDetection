# scripts/finetune_columbia_head_fixed.py
import os, sys, torch, numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torch.nn as nn

# ---- USER CONFIG ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_IN = "models/vgg16_baseline.pt"
MODEL_OUT = "models/vgg16_columbia_finetuned.pt"
ELA_BASE = "data/processed/ELA_columbia"
BATCH = 16
EPOCHS = 4
LR = 5e-4
NUM_WORKERS = 2
FALLBACK_UNFREEZE_LAST_LINEAR = 2
# ----------------------

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class ELADataset(Dataset):
    def __init__(self, auth_dir, tamp_dir, transform=tf):
        self.items=[]
        for f in sorted(os.listdir(auth_dir)):
            if f.lower().endswith(".jpg"):
                self.items.append((os.path.join(auth_dir,f), 0))
        for f in sorted(os.listdir(tamp_dir)):
            if f.lower().endswith(".jpg"):
                self.items.append((os.path.join(tamp_dir,f), 1))
        self.tf = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p,l = self.items[idx]
        img = Image.open(p).convert("RGB")
        return self.tf(img), l


# ------------------------------------------------------------
# ✅ FIXED CHECKPOINT LOADING (robust, correct syntax)
# ------------------------------------------------------------
def try_load_state(model, path):
    state = torch.load(path, map_location=DEVICE)

    # 1) Try direct load
    try:
        model.load_state_dict(state)
        print("Loaded checkpoint directly.")
        return
    except Exception:
        print("Direct load failed → trying prefix removal…")

    # 2) Remove “model.” prefix if it exists
    stripped = {
        (k[len("model."): ] if k.startswith("model.") else k): v
        for k, v in state.items()
    }

    try:
        model.load_state_dict(stripped, strict=False)
        print("Loaded after stripping prefix.")
        return
    except Exception:
        print("Prefix strip also failed → trying strict=False fallback.")

    # 3) Try best-effort loading
    model.load_state_dict(state, strict=False)
    print("Loaded checkpoint with strict=False fallback.")


# ------------------------------------------------------------
# UNFREEZE LOGIC
# ------------------------------------------------------------
def unfreeze_classifier_or_head(model, fallback_linear=FALLBACK_UNFREEZE_LAST_LINEAR):
    unfrozen = []

    if hasattr(model, "classifier"):
        for n, p in model.classifier.named_parameters():
            p.requires_grad = True
            unfrozen.append(("classifier."+n, p))
        return unfrozen

    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules.append((name, module))

    if linear_modules:
        for name, module in linear_modules[-fallback_linear:]:
            for n, p in module.named_parameters():
                p.requires_grad = True
                unfrozen.append((f"{name}.{n}", p))
        return unfrozen

    return unfrozen


def print_trainable_info(model):
    trainable = [(n,p.shape) for n,p in model.named_parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for n,p in model.named_parameters() if p.requires_grad)
    total_params = sum(p.numel() for n,p in model.named_parameters())
    print(f"Trainable params: {len(trainable)} tensors, {total_trainable} elements. Total params: {total_params}")
    for i,(n,shape) in enumerate(trainable[:20]):
        print("  ", i, n, shape)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    if not os.path.isdir(ELA_BASE):
        print("ERROR: ELA_BASE directory missing:", ELA_BASE)
        sys.exit(1)

    dataset = ELADataset(os.path.join(ELA_BASE,"authentic"),
                         os.path.join(ELA_BASE,"tampered"))

    if len(dataset) == 0:
        print("No ELA images found inside authentic/ and tampered/")
        sys.exit(1)

    n = len(dataset)
    idx = np.arange(n); np.random.seed(42); np.random.shuffle(idx)
    train_idx, val_idx = idx[:int(0.8*n)], idx[int(0.8*n):]

    tr = DataLoader(Subset(dataset, train_idx),
                    batch_size=BATCH, shuffle=True,
                    num_workers=NUM_WORKERS)

    vl = DataLoader(Subset(dataset, val_idx),
                    batch_size=BATCH, shuffle=False,
                    num_workers=NUM_WORKERS)

    # Build model
    from src.model_vgg import build_vgg16
    model = build_vgg16(num_classes=2).to(DEVICE)

    # Load CASIA-trained checkpoint safely
    if os.path.exists(MODEL_IN):
        print("Loading pretrained CASIA checkpoint:", MODEL_IN)
        try_load_state(model, MODEL_IN)
    else:
        print("WARNING: Pretrained model NOT FOUND → using imagenet initialization.")

    # Freeze all params
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze the classifier/head
    unfrozen = unfreeze_classifier_or_head(model)
    if not unfrozen:
        print("⚠️ No layers unfrozen — fallback unfreezing last linear layers")
        unfrozen = unfreeze_classifier_or_head(model)

    print_trainable_info(model)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        print("ERROR: No trainable parameters after all attempts.")
        sys.exit(1)

    opt = torch.optim.Adam(trainable_params, lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    # ---------------- TRAIN LOOP ----------------
    for ep in range(EPOCHS):
        model.train()
        loss_sum = 0; count = 0
        for x,y in tr:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out,y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()*x.size(0)
            count += x.size(0)
        print(f"Epoch {ep+1}/{EPOCHS} train_loss={loss_sum/count:.4f}")

        # ---- VAL ----
        model.eval()
        correct=0; total=0
        with torch.no_grad():
            for x,y in vl:
                x,y = x.to(DEVICE), y.to(DEVICE)
                pred = torch.argmax(model(x),1)
                correct += (pred==y).sum().item()
                total += y.size(0)

        acc = correct/total
        print(f"Epoch {ep+1} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_OUT)
            print("Saved best:", MODEL_OUT)

    print("FINISHED. Best val accuracy =", best_acc)


if __name__ == "__main__":
    main()
