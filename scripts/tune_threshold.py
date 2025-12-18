# scripts/tune_threshold.py
import numpy as np
data = np.load("results/eval_columbia/columbia_val_probs.npz")
y_true = data['y_true']; y_prob = data['y_prob']
from sklearn.metrics import f1_score, precision_recall_curve
ths = np.linspace(0.0,1.0,201)
best_t, best_f1 = 0.5, 0.0
for t in ths:
    pred = (y_prob >= t).astype(int)
    f1 = f1_score(y_true, pred, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1; best_t = t
print("Best threshold:", best_t, "F1:", best_f1)
# save
np.savez("results/eval_columbia/threshold_tuning.npz", best_t=best_t, best_f1=best_f1)
