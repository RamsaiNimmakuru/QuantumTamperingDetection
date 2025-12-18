# src/trainer_quantum_finetune_staged.py
import os, json, time
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from src.model_quantum import HybridQuantumCNN
from src.trainer_quantum import QuantumDataset  # assumes this yields (img_tensor, label)
import numpy as np

def get_sampler_if_needed(dataset):
    # build label list
    labels = []
    for _, y in dataset:
        labels.append(int(y))
    labels = np.array(labels)
    class_sample_count = np.array([ (labels==0).sum(), (labels==1).sum() ])
    if class_sample_count.min() == 0:
        return None
    weights = 1.0 / class_sample_count
    samples_weights = np.array([weights[y] for y in labels])
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    return sampler

def evaluate(model, loader, device):
    model.eval()
    correct = 0; total = 0; losses = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    preds_all = []; probs_all = []; labels_all = []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            losses += loss.item()
            pred = torch.argmax(out, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            probs_all += torch.softmax(out, dim=1)[:,1].cpu().tolist()
            preds_all += pred.cpu().tolist()
            labels_all += y.cpu().tolist()
    if total == 0:
        return {"loss": None, "acc": 0.0, "total": 0}
    return {"loss": losses/len(loader), "acc": correct/total, "total": total,
            "preds": preds_all, "probs": probs_all, "labels": labels_all}

def fine_tune_staged(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = HybridQuantumCNN().to(device)

    # load pretrained
    pre = cfg.get("pretrained_model", "")
    if pre and os.path.exists(pre):
        model.load_state_dict(torch.load(pre, map_location=device))
        print("✅ Loaded pretrained model:", pre)
    else:
        print("⚠️ Pretrained model not found or not specified:", pre)

    # datasets
    train_ds = QuantumDataset(cfg["train_dir"])
    val_ds   = QuantumDataset(cfg["val_dir"])

    # sampler if imbalance
    sampler = get_sampler_if_needed(train_ds)
    if sampler is not None:
        print("Using WeightedRandomSampler for class balance.")
        train_loader = DataLoader(train_ds, batch_size=cfg["batch"], sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch"], shuffle=False, num_workers=2)

    # metrics bookkeeping
    history = {"train_loss":[], "val_loss":[], "val_acc":[]}
    best_val_acc = 0.0
    best_model_path = os.path.join(cfg["model_dir"], "quantum_cnn_finetuned_staged.pt")
    os.makedirs(cfg["model_dir"], exist_ok=True)
    os.makedirs(cfg["result_dir"], exist_ok=True)

    # STAGE 1: train classifier head only
    print("\n=== Stage 1: Train classifier head only ===")
    for p in model.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=float(cfg.get("lr_stage1", 1e-4)), weight_decay=1e-6)
    criterion = torch.nn.CrossEntropyLoss()
    epochs1 = int(cfg.get("epochs_stage1", 3))

    for ep in range(epochs1):
        model.train(); tloss=0.0; it=0
        for x,y in tqdm(train_loader, desc=f"Stage1 Epoch {ep+1}/{epochs1}"):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out,y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tloss += loss.item(); it+=1
        tloss /= max(1,it)
        val = evaluate(model, val_loader, device)
        history["train_loss"].append(tloss)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        print(f"Stage1 Epoch {ep+1}: Train={tloss:.4f} Val={val['loss']:.4f} Acc={val['acc']*100:.2f}%")
        # checkpoint
        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            torch.save(model.state_dict(), best_model_path)
            print("🔖 Saved best model (stage1).")

    # STAGE 2: unfreeze quantum layer + classifier, train
    print("\n=== Stage 2: Train quantum layer + classifier ===")
    for p in model.feature_extractor.parameters(): p.requires_grad = False
    for p in model.quantum_layer.parameters(): p.requires_grad = True
    for p in model.classifier.parameters(): p.requires_grad = True

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=float(cfg.get("lr_stage2", 1e-5)), weight_decay=1e-6)
    epochs2 = int(cfg.get("epochs_stage2", 15))
    patience = int(cfg.get("patience", 5))
    wait = 0

    for ep in range(epochs2):
        model.train(); tloss=0.0; it=0
        for x,y in tqdm(train_loader, desc=f"Stage2 Epoch {ep+1}/{epochs2}"):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out,y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tloss += loss.item(); it+=1
        tloss /= max(1,it)
        val = evaluate(model, val_loader, device)
        history["train_loss"].append(tloss)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        print(f"Stage2 Epoch {ep+1}: Train={tloss:.4f} Val={val['loss']:.4f} Acc={val['acc']*100:.2f}%")
        # checkpoint + early stop
        if val["acc"] > best_val_acc + 1e-6:
            best_val_acc = val["acc"]
            torch.save(model.state_dict(), best_model_path)
            wait = 0
            print("🔖 Saved best model (stage2).")
        else:
            wait += 1
            if wait >= patience:
                print("⏱ Early stopping triggered.")
                break

    # STAGE 3: optional unfreeze last CNN block (if requested)
    if cfg.get("unfreeze_last_block", False):
        print("\n=== Stage 3: Unfreeze last CNN block and fine-tune ===")
        # Example for VGG: unfreeze features[24:] — adapt if architecture differs
        for name, p in model.feature_extractor.named_parameters():
            if any(part in name for part in cfg.get("last_block_names", ["features.24","features.26","features.28"])):
                p.requires_grad = True
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=float(cfg.get("lr_stage3", 1e-6)), weight_decay=1e-6)
        epochs3 = int(cfg.get("epochs_stage3", 5))
        for ep in range(epochs3):
            model.train(); tloss=0.0; it=0
            for x,y in tqdm(train_loader, desc=f"Stage3 Epoch {ep+1}/{epochs3}"):
                x,y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model(x)
                loss = criterion(out,y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                tloss += loss.item(); it+=1
            tloss /= max(1,it)
            val = evaluate(model, val_loader, device)
            history["train_loss"].append(tloss)
            history["val_loss"].append(val["loss"])
            history["val_acc"].append(val["acc"])
            print(f"Stage3 Epoch {ep+1}: Train={tloss:.4f} Val={val['loss']:.4f} Acc={val['acc']*100:.2f}%")
            if val["acc"] > best_val_acc:
                best_val_acc = val["acc"]
                torch.save(model.state_dict(), best_model_path)
                print("🔖 Saved best model (stage3).")

    # Save history
    with open(os.path.join(cfg["result_dir"], "finetune_staged_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("\n✅ Fine-tuning complete. Best val acc:", best_val_acc)
    print("Saved best model at:", best_model_path)
    return history
