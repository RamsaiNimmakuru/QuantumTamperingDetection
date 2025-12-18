# scripts/generate_quantum_features_offline.py
import numpy as np, torch
from sklearn.decomposition import PCA
from src.quantum_utils import QuantumLayer
import os

# load classical features
train_feats = np.load("data/features/train_feats.npy")
val_feats   = np.load("data/features/val_feats.npy")
pca_dim = 6   # match n_qubits in quantum layer
pca = PCA(n_components=pca_dim)
pca.fit(train_feats)
train_p = pca.transform(train_feats)
val_p   = pca.transform(val_feats)

# quantum layer (cpu)
ql = QuantumLayer().cpu()
def batch_qfeat(arr):
    qlist=[]
    for v in arr:
        t = torch.tensor(v, dtype=torch.float32).unsqueeze(0)  # shape [1, pca_dim]
        with torch.no_grad():
            qout = ql(t).squeeze(0).cpu().numpy()
        qlist.append(qout)
    return np.array(qlist)

os.makedirs("data/qfeatures", exist_ok=True)
q_train = batch_qfeat(train_p)
q_val   = batch_qfeat(val_p)
np.save("data/qfeatures/q_train.npy", q_train)
np.save("data/qfeatures/q_val.npy", q_val)
print("Saved quantum features:", q_train.shape, q_val.shape)
