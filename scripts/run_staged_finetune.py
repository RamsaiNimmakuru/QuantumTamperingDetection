# scripts/run_staged_finetune.py
import os, yaml, json, torch
from src.model_quantum import HybridQuantumCNN
from src.trainer_quantum import QuantumDataset  # your dataset class (returns (img_tensor, label))
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np

def get_sampler_if_needed(dataset):
    labels = []
    for _, y in dataset:
        labels.append(int(y))
    labels = np.array(labels)
    c0 = (labels == 0).sum(); c1 = (labels == 1).sum()
    if min(c0, c1) == 0:
        return None
    weights = 1.0 / np.array([c0, c1])
    samples_weights = np.array([weights[y] for y in labels])
    return WeightedRandomSampler(samples_weights, len(samples_weights))

def evaluate(model, loader, device):
    model.eval()
    correct=0; total=0; losses=0.0
    crit = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out,y)
            losses += loss.item()
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return {"loss": losses / max(1, len(loader)), "acc": correct / max(1,total)}

def staged_finetune(cfg):
    # cfg: dictionary with entries described later
    device_gpu = "cuda" if torch.cuda.is_available() else "cpu"
    print("GPU device for non-quantum stages:", device_gpu)
    model = HybridQuantumCNN(n_qubits=cfg.get("n_qubits", 8))
    # Load pretrained model if provided
    pre = cfg.get("pretrained_model", "")
    if pre and os.path.exists(pre):
        model.load_state_dict(torch.load(pre, map_location='cpu'))
        print("Loaded pretrained:", pre)

    # Datasets & loaders (we construct both; for CPU quantum stage we'll recreate loaders to ensure CPU tensors)
    train_ds = QuantumDataset(cfg["train_dir"])
    val_ds = QuantumDataset(cfg["val_dir"])

    sampler = get_sampler_if_needed(train_ds)
    train_loader_gpu = DataLoader(train_ds, batch_size=cfg["batch_gpu"], sampler=(sampler if sampler is not None else None), shuffle=(sampler is None))
    val_loader_gpu = DataLoader(val_ds, batch_size=cfg["batch_gpu"], shuffle=False)

    # === Stage 1: Train classifier head on GPU (or CPU if no GPU) ===
    device = device_gpu
    model.to(device)
    # Freeze backbone + quantum layer; train projection + classifier
    for p in model.feature_extractor.parameters(): p.requires_grad=False
    for p in model.quantum_layer.parameters(): p.requires_grad=False
    for p in model.projection.parameters(): p.requires_grad=True
    for p in model.classifier.parameters(): p.requires_grad=True

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.get("lr_stage1", 1e-4), weight_decay=1e-6)
    crit = torch.nn.CrossEntropyLoss()
    for ep in range(cfg.get("epochs_stage1", 3)):
        model.train()
        total_loss = 0.0; it=0
        for x,y in tqdm(train_loader_gpu, desc=f"Stage1 Epoch {ep+1}"):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); it += 1
        val = evaluate(model, val_loader_gpu, device)
        print(f"Stage1 Ep{ep+1} -> TrainLoss:{total_loss/max(1,it):.4f} ValAcc:{val['acc']*100:.2f}%")

    # === Stage 2: Quantum training on CPU (move model to CPU) ===
    print("Moving model to CPU for quantum stage...")
    model.to("cpu")
    # Recreate loaders with reasonable CPU batch sizes
    train_loader_cpu = DataLoader(train_ds, batch_size=cfg.get("batch_cpu", 4), sampler=(sampler if sampler is not None else None), shuffle=(sampler is None))
    val_loader_cpu = DataLoader(val_ds, batch_size=cfg.get("batch_cpu", 4), shuffle=False)

    # Freeze backbone, unfreeze quantum + classifier + projection
    for p in model.feature_extractor.parameters(): p.requires_grad = False
    for p in model.quantum_layer.parameters(): p.requires_grad = True
    for p in model.projection.parameters(): p.requires_grad = True
    for p in model.classifier.parameters(): p.requires_grad = True

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.get("lr_stage2", 1e-5), weight_decay=1e-6)
    best_val_acc = 0.0
    patience = cfg.get("patience", 4); wait=0
    for ep in range(cfg.get("epochs_stage2", 12)):
        model.train()
        total_loss=0.0; it=0
        for x,y in tqdm(train_loader_cpu, desc=f"Stage2 (quantum) Ep{ep+1}"):
            # keep tensors on CPU
            x,y = x.to("cpu"), y.to("cpu")
            opt.zero_grad()
            out = model(x)  # quantum forward runs on CPU inside QuantumLayer
            loss = crit(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); it += 1
        val = evaluate(model, val_loader_cpu, "cpu")
        print(f"Stage2 Ep{ep+1} -> TrainLoss:{total_loss/max(1,it):.4f} ValAcc:{val['acc']*100:.2f}%")
        # checkpoint
        if val['acc'] > best_val_acc + 1e-6:
            best_val_acc = val['acc']
            torch.save(model.state_dict(), cfg.get("best_model_path","models/quantum_cnn_finetuned_staged.pt"))
            wait = 0
            print("Saved best CPU-quantum model.")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping quantum stage.")
                break

    # Optionally move back to GPU and unfreeze some CNN layers
    if cfg.get("do_stage3_gpu_finetune", False) and device_gpu != "cpu":
        print("Moving model back to GPU for final CNN fine-tune...")
        model.to(device_gpu)
        # unfreeze last conv block manually if necessary (names depend on backbone)
        for name,p in model.feature_extractor.named_parameters():
            # example: unfreeze last few layers; modify as per your backbone
            if any(k in name for k in ["28", "26", "24"]):
                p.requires_grad = True
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.get("lr_stage3", 1e-6), weight_decay=1e-6)
        for ep in range(cfg.get("epochs_stage3", 3)):
            model.train()
            total_loss=0.0; it=0
            for x,y in tqdm(train_loader_gpu, desc=f"Stage3 GPU Ep{ep+1}"):
                x,y = x.to(device_gpu), y.to(device_gpu)
                opt.zero_grad()
                out = model(x)
                loss = crit(out,y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item(); it += 1
            val = evaluate(model, val_loader_gpu, device_gpu)
            print(f"Stage3 Ep{ep+1} -> TrainLoss:{total_loss/max(1,it):.4f} ValAcc:{val['acc']*100:.2f}%")
    print("Staged fine-tune complete. Best val acc:", best_val_acc)
    return
