# scripts/eval_external_dataset.py
import os
import io
import torch
import numpy as np
from PIL import Image, ImageChops
from tqdm import tqdm
from pprint import pprint
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt

# --- Adjust these paths if your repo layout differs ---
MODEL_PATH = "models/vgg16_baseline.pt"
RESULTS_DIR = "results/eval_external"
os.makedirs(RESULTS_DIR, exist_ok=True)

# dataset folders (change if needed)
COLUMBIA_AUTH = "data/raw/Columbia/4cam_auth/4cam_auth"
COLUMBIA_TAMP = "data/raw/Columbia/4cam_splc/4cam_splc"
BSD_AUTH = "data/raw/BSD500/images/test"           # proxy authentic
BSD_TAMP = "data/raw/BSD500/ground_truth/test"     # proxy tampered (edges/gt)

# --- Model import (expects src/model_vgg.py to expose build_vgg16) ---
from src.model_vgg import build_vgg16

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --------------------------
# ELA helper
# --------------------------
def generate_ela_pil(img_path, quality=95, size=(224,224)):
    """Return numpy array HxWx3 of ELA image resized to size."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to open image {img_path}: {e}")
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality)
    img_compressed = Image.open(buffer)
    ela = ImageChops.difference(img, img_compressed)
    extrema = ela.getextrema()
    scale = 1
    for channel in extrema:
        if channel[1] != 0:
            scale = max(scale, 255 // channel[1])
    ela = Image.eval(ela, lambda x: x * scale)
    ela = ela.resize(size)
    return np.array(ela)

# --------------------------
# Smart loader (handles 'model.' prefix mismatch)
# --------------------------
def smart_load_state_dict(model, ckpt_path, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)

    # unwrap if dict wrapper
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # If it's a whole model (rare), try returning it
    if not isinstance(ckpt, dict):
        print("Checkpoint is not a state-dict; attempting to use directly as model object.")
        return ckpt

    # try direct
    try:
        model.load_state_dict(ckpt)
        print("Loaded checkpoint directly (exact match).")
        return model
    except Exception:
        pass

    # prefix keys with 'model.'
    prefixed = {"model." + k: v for k, v in ckpt.items()}
    try:
        model.load_state_dict(prefixed, strict=False)
        print("Loaded checkpoint by prefixing keys with 'model.' (strict=False).")
        return model
    except Exception:
        pass

    # strip leading 'model.' from checkpoint keys
    stripped = {}
    for k, v in ckpt.items():
        if k.startswith("model."):
            stripped[k[len("model."):]] = v
        else:
            stripped[k] = v
    try:
        model.load_state_dict(stripped, strict=False)
        print("Loaded checkpoint by stripping leading 'model.' (strict=False).")
        return model
    except Exception as e:
        print("All automatic strategies failed. Final error follows.")
        raise RuntimeError("Couldn't adapt checkpoint keys to model.") from e

# --------------------------
# Evaluation helper
# --------------------------
transform = transforms.Compose([
    transforms.ToTensor()  # expects HxWx3 uint8 -> [0,1]
])

def eval_folder(model, folder, label):
    paths = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    ])
    y_true, y_pred, y_prob = [], [], []

    for p in tqdm(paths, desc=f"Eval {os.path.basename(folder)}"):
        try:
            ela = generate_ela_pil(p)
            x = transform(ela).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                prob = torch.softmax(logits, dim=1)[0][1].item()
            pred = 1 if prob > 0.5 else 0
            y_true.append(label)
            y_pred.append(pred)
            y_prob.append(prob)
        except Exception:
            # skip corrupt images
            continue

    return y_true, y_pred, y_prob

def compute_metrics(y_true, y_pred, y_prob, save_prefix):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    cm = confusion_matrix(y_true, y_pred)

    out = {
        "accuracy": float(acc),
        "precision": float(pre),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm.tolist()
    }

    # save confusion matrix image
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix: {save_prefix}")
    plt.colorbar()
    plt.xticks([0,1], ["auth","tamp"])
    plt.yticks([0,1], ["auth","tamp"])
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha='center', va='center', color='white' if v>cm.max()/2 else 'black')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(RESULTS_DIR, f"{save_prefix}_confusion.png")
    plt.tight_layout(); plt.savefig(cm_path); plt.close()

    # save ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.plot([0,1],[0,1],"--", linewidth=0.8)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC: {save_prefix}")
        plt.legend(loc="lower right")
        roc_path = os.path.join(RESULTS_DIR, f"{save_prefix}_roc.png")
        plt.tight_layout(); plt.savefig(roc_path); plt.close()
    except Exception:
        roc_path = None

    # save metrics json-like text
    metrics_path = os.path.join(RESULTS_DIR, f"{save_prefix}_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(str(out))

    return out, cm_path, roc_path, metrics_path

# --------------------------
# Main
# --------------------------
def main():
    # build model and smart-load weights
    model = build_vgg16(num_classes=2)
    model.to(device)
    try:
        model = smart_load_state_dict(model, MODEL_PATH, map_location=device)
    except Exception as e:
        print("Failed to load model checkpoint:", e)
        return
    model.eval()

    # Columbia eval
    print("\n=== EVALUATING COLUMBIA ===")
    yt1, yp1, yp_prob1 = eval_folder(model, COLUMBIA_AUTH, 0)
    yt2, yp2, yp_prob2 = eval_folder(model, COLUMBIA_TAMP, 1)

    y_true = yt1 + yt2
    y_pred = yp1 + yp2
    y_prob = yp_prob1 + yp_prob2

    col_metrics, col_cm, col_roc, col_metrics_path = compute_metrics(y_true, y_pred, y_prob, "columbia")
    print("\nColumbia metrics:")
    pprint(col_metrics)
    print("Saved:", col_cm, col_roc, col_metrics_path)

    # BSD eval
    print("\n=== EVALUATING BSD500 (proxy test) ===")
    bt1, bp1, bp_prob1 = eval_folder(model, BSD_AUTH, 0)
    bt2, bp2, bp_prob2 = eval_folder(model, BSD_TAMP, 1)

    y_true_b = bt1 + bt2
    y_pred_b = bp1 + bp2
    y_prob_b = bp_prob1 + bp_prob2

    bsd_metrics, bsd_cm, bsd_roc, bsd_metrics_path = compute_metrics(y_true_b, y_pred_b, y_prob_b, "bsd500")
    print("\nBSD500 metrics:")
    pprint(bsd_metrics)
    print("Saved:", bsd_cm, bsd_roc, bsd_metrics_path)

    print("\n✅ Evaluation finished. Results saved under:", RESULTS_DIR)

if __name__ == "__main__":
    main()
