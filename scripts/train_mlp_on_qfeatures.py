# scripts/train_mlp_on_qfeatures.py
import numpy as np, torch, os
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

q_train = np.load("data/qfeatures/q_train.npy")
y_train = np.load("data/features/train_labels.npy")
q_val = np.load("data/qfeatures/q_val.npy")
y_val = np.load("data/features/val_labels.npy")

device = "cuda" if torch.cuda.is_available() else "cpu"
X_train = torch.tensor(q_train, dtype=torch.float32)
Y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(q_val, dtype=torch.float32)
Y_val = torch.tensor(y_val, dtype=torch.long)

train_ds = TensorDataset(X_train, Y_train)
val_ds = TensorDataset(X_val, Y_val)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

model = nn.Sequential(
    nn.Linear(q_train.shape[1], 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

for ep in range(40):
    model.train(); tloss=0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out,yb)
        loss.backward(); opt.step()
        tloss += loss.item()
    # val
    model.eval()
    correct=0; total=0
    with torch.no_grad():
        for xb,yb in val_loader:
            xb,yb = xb.to(device), yb.to(device)
            out = model(xb)
            pred = out.argmax(1)
            correct += (pred==yb).sum().item(); total += yb.size(0)
    acc = correct/total
    if ep%5==0:
        print(f"Epoch {ep}: Train loss {tloss/len(train_loader):.4f} Val acc {acc*100:.2f}%")

torch.save(model.state_dict(), "models/mlp_qfeatures.pt")
print("Saved MLP on quantum features.")
