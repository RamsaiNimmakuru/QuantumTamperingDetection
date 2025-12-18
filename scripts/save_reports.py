# scripts/save_reports.py
import os, json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

os.makedirs("results", exist_ok=True)
y_true = np.load("results/y_true.npy")
y_scores = np.load("results/y_scores.npy")
y_pred = (y_scores >= 0.5).astype(int)

# classification report
rep = classification_report(y_true, y_pred, target_names=["Real","Tampered"])
with open("results/classification_report.txt", "w") as f:
    f.write(rep)
print(rep)

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(4,4)); plt.imshow(cm, cmap='Blues'); plt.title("Confusion Matrix"); plt.colorbar()
plt.xticks([0,1], ["Real","Tampered"]); plt.yticks([0,1], ["Real","Tampered"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.savefig("results/confusion_matrix.png", dpi=150); plt.close()

# ROC
fpr, tpr, _ = roc_curve(y_true, y_scores); roc_auc = auc(fpr, tpr)
plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}"); plt.plot([0,1],[0,1],'--'); plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.savefig("results/roc_curve.png", dpi=150); plt.close()

# PR
precision, recall, _ = precision_recall_curve(y_true, y_scores); ap = average_precision_score(y_true, y_scores)
plt.figure(); plt.plot(recall, precision, label=f"AP={ap:.4f}"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall"); plt.legend(); plt.savefig("results/pr_curve.png", dpi=150); plt.close()

# calibration
prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10)
plt.figure(); plt.plot(prob_pred, prob_true, marker='o'); plt.plot([0,1],[0,1],'--', color='gray'); plt.xlabel("Mean predicted prob"); plt.ylabel("Fraction of positives"); plt.title("Calibration"); plt.savefig("results/calibration.png", dpi=150); plt.close()

# summary
summary = {"roc_auc": float(roc_auc), "average_precision": float(ap), "n": int(len(y_true))}
with open("results/summary_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved metrics in results/")
