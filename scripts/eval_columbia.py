# scripts/eval_columbia.py
import os, torch, numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from src.model_vgg import build_vgg16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/vgg16_baseline.pt"
ELA_BASE = "data/processed/ELA_columbia"  # authentic/, tampered/

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def gather_files(dir_auth, dir_tamp):
    a_files = sorted([os.path.join(dir_auth,f) for f in os.listdir(dir_auth) if f.lower().endswith(".jpg")])
    t_files = sorted([os.path.join(dir_tamp,f) for f in os.listdir(dir_tamp) if f.lower().endswith(".jpg")])
    files = a_files + t_files
    labels = [0]*len(a_files) + [1]*len(t_files)
    return files, labels

files, labels = gather_files(os.path.join(ELA_BASE,"authentic"), os.path.join(ELA_BASE,"tampered"))
print("Images:", len(files))
model = build_vgg16(num_classes=2)
state = torch.load(MODEL_PATH, map_location=DEVICE)
# try load with or without "model." prefix
try:
    model.load_state_dict(state)
except:
    try:
        model.load_state_dict({("model."+k if not k.startswith("model.") else k):v for k,v in state.items()}, strict=False)
    except:
        model.load_state_dict(state, strict=False)
model.to(DEVICE).eval()

probs = []
preds = []
batch = []
batch_idx = []
with torch.no_grad():
    for i, p in enumerate(files):
        img = Image.open(p).convert("RGB")
        x = tf(img).unsqueeze(0).to(DEVICE)
        out = torch.softmax(model(x), dim=1)[:,1].item()
        probs.append(out)
        preds.append(1 if out>=0.5 else 0)

y_true = np.array(labels)
y_prob = np.array(probs)
y_pred = np.array(preds)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
try:
    auc = roc_auc_score(y_true, y_prob)
except:
    auc = float("nan")
cm = confusion_matrix(y_true, y_pred).tolist()

out_dir = "results/eval_columbia"
os.makedirs(out_dir, exist_ok=True)
np.savez(os.path.join(out_dir,"columbia_val_probs.npz"), y_true=y_true, y_prob=y_prob)
with open(os.path.join(out_dir,"columbia_metrics.txt"), "w") as f:
    f.write(str({"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"auc":auc,"confusion_matrix":cm}))
print("Metrics saved to", out_dir)
# plot ROC
from sklearn.metrics import roc_curve, auc as aucfunc
fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure(); plt.plot(fpr,tpr,label=f"AUC={auc:.3f}"); plt.plot([0,1],[0,1],"--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
plt.savefig(os.path.join(out_dir,"columbia_roc.png"))
# confusion matrix plot
import seaborn as sns
plt.figure(); sns.heatmap(cm, annot=True, fmt="d"); plt.savefig(os.path.join(out_dir,"columbia_confusion.png"))
print("Done. Accuracy:", acc, "F1:", f1, "AUC:", auc)
