import os, json, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from src.model_quantum import HybridQuantumCNN
from src.trainer_quantum import QuantumDataset

def fine_tune(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridQuantumCNN().to(device)

    # Load pretrained weights
    if os.path.exists(cfg["pretrained_model"]):
        model.load_state_dict(torch.load(cfg["pretrained_model"], map_location=device))
        print("✅ Loaded pre-trained model for fine-tuning")

    # Freeze lower CNN layers; train only quantum + classifier
    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=float(cfg["lr"]))
    criterion = torch.nn.CrossEntropyLoss()

    train_ds = QuantumDataset(cfg["train_dir"])
    val_ds   = QuantumDataset(cfg["val_dir"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch"], shuffle=False)

    metrics = {"train_loss": [], "val_loss": [], "acc": []}

    for ep in range(cfg["epochs"]):
        model.train(); total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Fine-tune Epoch {ep+1}/{cfg['epochs']}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); opt.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)

        # Validation
        model.eval(); val_loss = 0; correct = 0; total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                preds = torch.argmax(out, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        metrics["train_loss"].append(total_loss)
        metrics["val_loss"].append(val_loss / len(val_loader))
        metrics["acc"].append(acc)
        print(f"Epoch {ep+1}: Train={total_loss:.4f}  Val={val_loss:.4f}  Acc={acc*100:.2f}%")

    os.makedirs(cfg["result_dir"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg["model_dir"], "quantum_cnn_finetuned.pt"))
    with open(os.path.join(cfg["result_dir"], "finetune_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("✅ Fine-tuned model saved: models/quantum_cnn_finetuned.pt")
    return metrics
