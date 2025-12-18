# scripts/compare_casia_columbia_fixed.py
import os, torch, sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# ---- change this if your model_builder function is different ----
from src.model_vgg import build_vgg16
# ------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/vgg16_baseline.pt"   # your CASIA-trained checkpoint
BATCH = 32

# ---------- Dataset ----------
class ELADataset(Dataset):
    def __init__(self, auth, tamp, tf):
        self.items = []
        for f in sorted(os.listdir(auth)):
            if f.lower().endswith(".jpg"):
                self.items.append((os.path.join(auth, f), 0))
        for f in sorted(os.listdir(tamp)):
            if f.lower().endswith(".jpg"):
                self.items.append((os.path.join(tamp, f), 1))
        self.tf = tf

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        p, lbl = self.items[idx]
        img = Image.open(p).convert("RGB")
        return self.tf(img), lbl

# ---------- Transforms ----------
tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------- robust loader ----------
def load_checkpoint_flexible(model, ckpt_path, device=DEVICE):
    """
    Try multiple ways to load checkpoint:
      1) direct load
      2) strip 'model.' prefix from checkpoint keys
      3) add 'model.' prefix to checkpoint keys
      4) strict=False fallback
    Returns tuple (model, method_str)
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    # if state is a dict with 'state_dict' key (common), extract it
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    # 1) try direct
    try:
        model.load_state_dict(state_dict)
        return model, "direct"
    except Exception as e:
        # continue
        pass

    # 2) try stripping 'model.' prefix if present in keys
    stripped = { (k[len("model."): ] if k.startswith("model.") else k): v for k,v in state_dict.items() }
    try:
        model.load_state_dict(stripped)
        return model, "stripped_model_prefix_direct"
    except Exception:
        pass

    # 3) try adding 'model.' prefix to keys (if model expects 'model.xxx')
    added = { (("model."+k) if not k.startswith("model.") else k): v for k,v in state_dict.items() }
    try:
        model.load_state_dict(added, strict=False)
        return model, "added_model_prefix_strictFalse"
    except Exception:
        pass

    # 4) try strict=False with original
    try:
        model.load_state_dict(state_dict, strict=False)
        return model, "original_strictFalse"
    except Exception:
        pass

    # 5) try strict=False with stripped
    try:
        model.load_state_dict(stripped, strict=False)
        return model, "stripped_strictFalse"
    except Exception as final_e:
        # If we still fail, raise with helpful diagnostics
        ckpt_keys = list(state_dict.keys())[:20]
        model_keys = list(dict(model.named_parameters()).keys())[:20]
        msg = (
            "Failed to load checkpoint. Quick diagnostics:\n"
            f" sample ckpt keys (first 20): {ckpt_keys}\n"
            f" sample model param keys (first 20): {model_keys}\n"
            f" final loader exception: {final_e}\n"
            "Try inspecting key naming conventions (e.g. 'model.' prefix / wrapper)."
        )
        raise RuntimeError(msg)

# ---------- Evaluation Function ----------
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE)
            out = model(x)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(pred)
            labels.extend(y.numpy())

    acc = accuracy_score(labels, preds) if len(labels)>0 else 0.0
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, preds)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(labels, preds) if len(labels)>0 else None
    return acc, prec, rec, f1, auc, cm

# ---------- MAIN ----------
def main():
    # build model
    model = build_vgg16(num_classes=2)
    model = model.to(DEVICE)

    # load checkpoint robustly
    model, method = load_checkpoint_flexible(model, MODEL_PATH, device=DEVICE)
    print(f"Checkpoint load method used: {method}")

    # CASIA: use your CASIA ELA train/val or a subset path (adjust paths if different)
    casia_auth = "data/processed/ELA/train/authentic"
    casia_tamp = "data/processed/ELA/train/tampered"
    if not (os.path.isdir(casia_auth) and os.path.isdir(casia_tamp)):
        print("WARNING: CASIA ELA paths not found:", casia_auth, casia_tamp)
    else:
        casia_ds = ELADataset(casia_auth, casia_tamp, tf)
        casia_loader = DataLoader(casia_ds, batch_size=BATCH, shuffle=False, num_workers=4)
        casia_metrics = evaluate(model, casia_loader)
        print("\nCASIA metrics: Acc {:.4f}, Prec {:.4f}, Rec {:.4f}, F1 {:.4f}, AUC {:.4f}".format(*casia_metrics[:-1], casia_metrics[4]))

    # Columbia ELA
    col_auth = "data/processed/ELA_columbia/authentic"
    col_tamp = "data/processed/ELA_columbia/tampered"
    if not (os.path.isdir(col_auth) and os.path.isdir(col_tamp)):
        print("ERROR: Columbia ELA folders not found:", col_auth, col_tamp)
        sys.exit(1)

    col_ds = ELADataset(col_auth, col_tamp, tf)
    col_loader = DataLoader(col_ds, batch_size=BATCH, shuffle=False, num_workers=4)
    col_metrics = evaluate(model, col_loader)

    print("\n=== Columbia metrics ===")
    print(f"Accuracy : {col_metrics[0]:.4f}")
    print(f"Precision: {col_metrics[1]:.4f}")
    print(f"Recall   : {col_metrics[2]:.4f}")
    print(f"F1       : {col_metrics[3]:.4f}")
    print(f"AUC      : {col_metrics[4]:.4f}")
    print(f"Confusion matrix:\n{col_metrics[5]}")

if __name__ == "__main__":
    main()
