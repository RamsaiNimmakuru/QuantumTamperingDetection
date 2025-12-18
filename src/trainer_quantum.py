import os, json, torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from src.model_quantum import HybridQuantumCNN

class QuantumDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = []
        for cls in ["authentic", "tampered"]:
            folder = os.path.join(root, cls)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.endswith((".jpg",".png",".tif")):
                        self.files.append((os.path.join(folder,f), 0 if cls=="authentic" else 1))
        self.transform = transform or transforms.Compose([
            transforms.Resize((128,128)), transforms.ToTensor()
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), torch.tensor(label)

def fit_hybrid(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridQuantumCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    criterion = torch.nn.CrossEntropyLoss()

    train_ds = QuantumDataset(cfg["train_dir"])
    val_ds = QuantumDataset(cfg["val_dir"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch"], shuffle=False)

    metrics = {"train_loss": [], "val_loss": [], "acc": []}

    for ep in range(cfg["epochs"]):
        model.train(); total_loss=0
        for x,y in tqdm(train_loader, desc=f"Epoch {ep+1}/{cfg['epochs']}"):
            x,y=x.to(device),y.to(device)
            opt.zero_grad()
            out=model(x)
            loss=criterion(out,y)
            loss.backward(); opt.step()
            total_loss+=loss.item()
        total_loss/=len(train_loader)

        model.eval(); val_loss=0; correct=0; total=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device)
                out=model(x)
                loss=criterion(out,y)
                val_loss+=loss.item()
                preds=torch.argmax(out,dim=1)
                correct+=(preds==y).sum().item()
                total+=y.size(0)
        acc=correct/total
        metrics["train_loss"].append(total_loss)
        metrics["val_loss"].append(val_loss/len(val_loader))
        metrics["acc"].append(acc)
        print(f"Epoch {ep+1}: Train={total_loss:.3f} Val={val_loss:.3f} Acc={acc*100:.2f}%")

    os.makedirs(cfg["result_dir"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg["model_dir"], "quantum_cnn.pt"))
    with open(os.path.join(cfg["result_dir"], "hybrid_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("✅ Hybrid model saved:", os.path.join(cfg["model_dir"], "quantum_cnn.pt"))
    return metrics
