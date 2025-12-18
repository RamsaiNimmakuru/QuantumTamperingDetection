# scripts/train_multistream_prototype.py
import os, io, random, argparse, json
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from albumentations import Compose, RandomResizedCrop, HorizontalFlip, ColorJitter, Normalize
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, recall_score
from tqdm import tqdm

from src.model_multistream import MultiStreamFusion

def create_ela_image(pil_img: Image.Image, quality=90, scale=10):
    buffer = io.BytesIO()
    pil_img.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert('RGB')
    ela = ImageChops.difference(pil_img.convert('RGB'), recompressed)
    extrema = ela.getextrema()
    max_diff = max([e[1] for e in extrema]) or 1
    factor = scale * (255.0 / max_diff)
    ela = ImageEnhance.Brightness(ela).enhance(factor)
    return ela

class ELAImageFolder(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.files = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        label = self.labels[idx]
        img = Image.open(p).convert('RGB')
        ela = create_ela_image(img, quality=90, scale=10)
        img_np = np.array(img); ela_np = np.array(ela)
        joint = np.concatenate([img_np, ela_np], axis=2)
        if self.transform is not None:
            aug = self.transform(image=joint)
            aug_joint = aug['image']
            rgb_t = aug_joint[:3,...].float()
            ela_t = aug_joint[3:6,...].float()
        else:
            toT = T.ToTensor()
            rgb_t = toT(img)
            ela_t = toT(ela)
        rgb_np = np.transpose(rgb_t.numpy(), (1,2,0)) * 255.0
        try:
            from scipy.ndimage import gaussian_filter
            blurred = gaussian_filter(rgb_np, sigma=(1,1,0))
        except Exception:
            blurred = rgb_np
        residual = rgb_np - blurred
        residual = np.clip(residual/255.0, -1.0, 1.0)
        residual_t = torch.from_numpy(np.transpose(residual, (2,0,1))).float()
        return rgb_t, ela_t, residual_t, torch.tensor(label, dtype=torch.long)

def build_file_list(data_dir):
    classes = ['Real','Tampered']
    files=[]; labels=[]
    for i,c in enumerate(classes):
        p = os.path.join(data_dir, c)
        for f in glob(os.path.join(p, '*.*')):
            files.append(f); labels.append(i)
    return files, labels

def compute_metrics(y_true, y_pred, y_score):
    acc = (np.array(y_true)==np.array(y_pred)).mean()
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score) if len(set(y_true))>1 else 0.0
    rec_tampered = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    return dict(acc=float(acc), f1=float(f1), auc=float(auc), recall_tampered=float(rec_tampered))

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    files, labels = build_file_list(args.data_dir)
    print("Total files:", len(files))
    zipped = list(zip(files, labels)); random.seed(42); random.shuffle(zipped)
    files, labels = zip(*zipped)
    n = len(files); ntrain = int(0.8*n)
    train_files = list(files[:ntrain]); train_labels = list(labels[:ntrain])
    val_files = list(files[ntrain:]); val_labels = list(labels[ntrain:])
    train_tf = Compose([
        RandomResizedCrop(224,224, scale=(0.6,1.0), p=0.9),
        HorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    val_tf = Compose([
        Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    train_ds = ELAImageFolder(train_files, train_labels, transform=train_tf)
    val_ds = ELAImageFolder(val_files, val_labels, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = MultiStreamFusion(pretrained_rgb=True).to(device)
    for p in model.rgb.features.parameters():
        p.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_recall = 0.0
    history = {"train_loss":[], "val_loss":[], "val_recall":[], "val_acc":[]}
    for epoch in range(1, args.epochs+1):
        model.train(); running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for rgb, ela, res, label in pbar:
            rgb = rgb.to(device); ela = ela.to(device); res = res.to(device); label = label.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(rgb, ela, res)
                loss = criterion(out, label)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running_loss += float(loss.item()) * rgb.size(0)
        train_loss = running_loss / len(train_ds); history["train_loss"].append(train_loss)
        model.eval(); y_true=[]; y_pred=[]; y_score=[]; val_loss = 0.0
        with torch.no_grad():
            for rgb, ela, res, label in val_loader:
                rgb = rgb.to(device); ela = ela.to(device); res = res.to(device); label = label.to(device)
                out = model(rgb, ela, res)
                loss = criterion(out, label)
                val_loss += float(loss.item()) * rgb.size(0)
                probs = torch.softmax(out, dim=1)[:,1].cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                y_true.extend(label.cpu().numpy().tolist())
                y_pred.extend(preds.tolist())
                y_score.extend(probs.tolist())
        val_loss = val_loss / len(val_ds)
        metrics = compute_metrics(y_true, y_pred, y_score)
        history["val_loss"].append(val_loss); history["val_recall"].append(metrics["recall_tampered"]); history["val_acc"].append(metrics["acc"])
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={metrics['acc']:.4f} val_recall={metrics['recall_tampered']:.4f} f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}")
        if metrics["recall_tampered"] > best_recall:
            best_recall = metrics["recall_tampered"]
            ckpt = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "metrics": metrics}
            torch.save(ckpt, f"models/multistream_best.pt")
            print("Saved best model -> models/multistream_best.pt")
        if epoch == args.unfreeze_epoch:
            print("Unfreezing RGB last layers")
            for name, param in model.rgb.features.named_parameters():
                param.requires_grad = True
    torch.save({"model_state": model.state_dict()}, "models/multistream_last.pt")
    import numpy as np
    np.save("results/y_true.npy", np.array(y_true)); np.save("results/y_pred.npy", np.array(y_pred)); np.save("results/y_scores.npy", np.array(y_score))
    with open("results/classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=["Real","Tampered"]))
    with open("results/metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Training finished. Saved results in results/ and models/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data", help="root data dir with Real/ Tampered subfolders")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze-epoch", type=int, default=3)
    args = parser.parse_args()
    train(args)
