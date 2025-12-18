import torch, json, os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from src.model_quantum import HybridQuantumCNN

# Paths
MODEL_PATH = "models/quantum_cnn_finetuned.pt"
DATA_DIR = "data/processed/ELA_finetune/val"
RESULT_DIR = "results/quantum_finetune/evaluation"
os.makedirs(RESULT_DIR, exist_ok=True)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = HybridQuantumCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Transform
tf = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# Collect predictions
y_true, y_pred, y_prob = [], [], []

for cls, label in [("authentic", 0), ("tampered", 1)]:
    folder = os.path.join(DATA_DIR, cls)
    for file in tqdm(os.listdir(folder), desc=f"Evaluating {cls}"):
        if file.endswith((".jpg",".png",".tif")):
            img = Image.open(os.path.join(folder, file)).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x)
                prob = torch.softmax(out, dim=1)[0][1].item()
                pred = torch.argmax(out, dim=1).item()
            y_true.append(label)
            y_pred.append(pred)
            y_prob.append(prob)

# Metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)
cm = confusion_matrix(y_true, y_pred)

# Save & print
results = {
    "Accuracy": acc * 100,
    "Precision": prec * 100,
    "Recall": rec * 100,
    "F1-Score": f1 * 100,
    "ROC-AUC": auc * 100
}

print("\n✅ Evaluation Metrics:")
for k,v in results.items():
    print(f"{k}: {v:.2f}%")

# Confusion Matrix Plot
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Authentic", "Tampered"],
            yticklabels=["Authentic", "Tampered"])
plt.title("Confusion Matrix – Fine-Tuned Quantum Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))

with open(os.path.join(RESULT_DIR, "eval_metrics.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n📊 Saved metrics → {RESULT_DIR}/eval_metrics.json")
