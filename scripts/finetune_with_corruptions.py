# scripts/finetune_with_corruptions.py
# (copy exactly — same as provided earlier)
import os
import argparse
import random
from collections import defaultdict
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, Resize, RandomCrop, HorizontalFlip, Normalize
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
from tqdm import tqdm
from src.model_multistream import MultiStreamFusion

def corrupt_jpeg(img, q=70):
    import io
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=q)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def corrupt_down_up(img, scale=0.5):
    w,h = img.size
    small = img.resize((max(1,int(w*scale)), max(1,int(h*scale))), Image.BILINEAR)
    return small.resize((w,h), Image.BILINEAR)

def corrupt_blur(img, r=1.0):
    return img.filter(ImageFilter.GaussianBlur(radius=r))

def corrupt_brightness_contrast(img, b=1.0, c=1.0):
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    return img

IMG_EXTS = ('.jpg','.jpeg','.png','.bmp','.tiff','.webp')
def is_image(f): return f.lower().endswith(IMG_EXTS)
def basename_no_ext(path): return os.path.splitext(os.path.basename(path))[0]
def find_all_images(root):
    files=[]
    for r,_,fn in os.walk(root):
        for f in fn:
            if is_image(f): files.append(os.path.join(r,f))
    return files
def build_pairs(rgb_root, ela_root):
    ela_files=[]
    for r,_,fn in os.walk(ela_root):
        for f in fn:
            if is_image(f):
                full=os.path.join(r,f)
                lower=full.lower()
                if any(k in lower for k in ['/auth','/real','/orig','/genuine','/authentic']): label=0
                elif any(k in lower for k in ['/tamper','/tampered','/fake','/forged','/splice','/tp','/splc']): label=1
                else:
                    parent=os.path.basename(os.path.dirname(full)).lower()
                    if 'auth' in parent or 'real' in parent or 'orig' in parent: label=0
                    elif 'tamper' in parent or 'tampered' in parent or 'tp' in parent or 'splc' in parent: label=1
                    else: label=None
                if label is not None: ela_files.append((full,label))
    rgb_files=find_all_images(rgb_root)
    idx=defaultdict(list)
    for p in rgb_files: idx[basename_no_ext(p)].append(p)
    pairs=[]
    skipped=0
    for ela_path,label in ela_files:
        base=basename_no_ext(ela_path)
        found=None
        if base in idx: found=idx[base][0]
        else:
            for k,lst in idx.items():
                if k in base or base in k:
                    found=lst[0]; break
        if not found:
            short=base
            for prefix in ['au_','tp_','au','tp','img_','img','pla','nat','sec','arc','ind','txt']:
                if short.startswith(prefix):
                    cand=short[len(prefix):]
                    if cand in idx:
                        found=idx[cand][0]; break
        if not found:
            for p in rgb_files:
                b=basename_no_ext(p)
                if base in b or b in base:
                    found=p; break
        if found: pairs.append((found, ela_path, label))
        else:
            skipped+=1
    print(f"Paired {len(pairs)} vs skipped {skipped}")
    return pairs

class PairedCorruptDataset(Dataset):
    def __init__(self,pairs,transform=None,ela_as_rgb=False,base_size=256,corrupt_prob=0.5,corruption_cfg=None):
        self.pairs=pairs
        self.transform=transform
        self.ela_as_rgb=ela_as_rgb
        self.base_size=base_size
        self.corrupt_prob=corrupt_prob
        self.corruption_cfg=corruption_cfg or {"jpeg":[(90,0.1),(70,0.2),(40,0.1)], "down":[(0.5,0.15),(0.25,0.05)], "blur":[(1.0,0.15),(2.0,0.05)], "brightness_contrast":[(0.9,1.0,0.1)]}
    def __len__(self): return len(self.pairs)
    def _apply_random_corruption(self, img):
        if random.random() < 0.33:
            options=self.corruption_cfg.get("jpeg",[])
            if options:
                qs=[q for q,p in options]; ps=[p for q,p in options]
                q=random.choices(qs, weights=ps, k=1)[0]
                return corrupt_jpeg(img, q=q)
        if random.random() < 0.25:
            options=self.corruption_cfg.get("down",[])
            if options:
                scales=[s for s,p in options]; ps=[p for s,p in options]
                s=random.choices(scales, weights=ps, k=1)[0]
                return corrupt_down_up(img, scale=s)
        if random.random() < 0.25:
            options=self.corruption_cfg.get("blur",[])
            if options:
                rs=[r for r,p in options]; ps=[p for r,p in options]
                r=random.choices(rs, weights=ps, k=1)[0]
                return corrupt_blur(img, r=r)
        if random.random() < 0.2:
            options=self.corruption_cfg.get("brightness_contrast",[])
            if options:
                b,c,p = random.choice(options)
                return corrupt_brightness_contrast(img, b=b, c=c)
        return img
    def __getitem__(self, idx):
        rgb_p, ela_p, label = self.pairs[idx]
        rgb = Image.open(rgb_p).convert('RGB')
        ela = Image.open(ela_p).convert('L') if not self.ela_as_rgb else Image.open(ela_p).convert('RGB')
        if random.random() < self.corrupt_prob:
            rgb = self._apply_random_corruption(rgb)
        rgb = rgb.resize((self.base_size,self.base_size), Image.BILINEAR)
        if ela.mode == 'L':
            ela_rgb = Image.merge('RGB',(ela,ela,ela)).resize((self.base_size,self.base_size), Image.BILINEAR)
        else:
            ela_rgb = ela.convert('RGB').resize((self.base_size,self.base_size), Image.BILINEAR)
        joint = np.concatenate([np.array(rgb), np.array(ela_rgb)], axis=2)
        if self.transform:
            aug = self.transform(image=joint); aug_joint = aug['image']
            rgb_t = aug_joint[:3,...].float(); ela_t = aug_joint[3:6,...].float()
        else:
            toT=T.ToTensor(); rgb_t=toT(rgb); ela_t=toT(ela_rgb)
        rgb_np_f = np.transpose(rgb_t.numpy(), (1,2,0)) * 255.0
        try:
            from scipy.ndimage import gaussian_filter
            blurred = gaussian_filter(rgb_np_f, sigma=(1,1,0))
        except Exception:
            blurred = rgb_np_f
        residual = rgb_np_f - blurred
        residual = np.clip(residual/255.0, -1.0, 1.0)
        residual_t = torch.from_numpy(np.transpose(residual,(2,0,1))).float()
        return rgb_t, ela_t, residual_t, torch.tensor(label, dtype=torch.long)
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pairs = build_pairs(args.rgb_root, args.ela_root)
    random.shuffle(pairs)
    ntrain = int(0.8*len(pairs))
    train_pairs = pairs[:ntrain]
    val_pairs = pairs[ntrain:]
    train_tf = Compose([ Resize(height=256,width=256,p=1.0), RandomCrop(height=224,width=224,p=1.0), HorizontalFlip(p=0.5), Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)), ToTensorV2() ])
    val_tf = Compose([ Resize(height=224,width=224,p=1.0), Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)), ToTensorV2() ])
    train_ds = PairedCorruptDataset(train_pairs, transform=train_tf, ela_as_rgb=args.ela_as_rgb, base_size=args.base_size, corrupt_prob=args.corrupt_prob)
    val_ds = PairedCorruptDataset(val_pairs, transform=val_tf, ela_as_rgb=args.ela_as_rgb, base_size=args.base_size, corrupt_prob=0.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = MultiStreamFusion(pretrained_rgb=False).to(device)
    ck = torch.load(args.ckpt, map_location='cpu')
    if 'model_state' in ck: model.load_state_dict(ck['model_state'])
    else: model.load_state_dict(ck)
    if args.freeze_backbone:
        for p in model.rgb.features.parameters(): p.requires_grad=False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_recall=0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        running=0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for rgb, ela, res, label in pbar:
            rgb=rgb.to(device); ela=ela.to(device); res=res.to(device); label=label.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(rgb, ela, res)
                loss = criterion(out, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())*rgb.size(0)
        train_loss = running/len(train_ds)
        model.eval()
        y_true=[]; y_pred=[]; y_score=[]
        with torch.no_grad():
            for rgb, ela, res, label in val_loader:
                rgb=rgb.to(device); ela=ela.to(device); res=res.to(device); label=label.to(device)
                out = model(rgb, ela, res)
                probs = torch.softmax(out, dim=1)[:,1].cpu().numpy()
                preds = (probs>=0.5).astype(int)
                y_true.extend(label.cpu().numpy().tolist())
                y_pred.extend(preds.tolist())
                y_score.extend(probs.tolist())
        try:
            from sklearn.metrics import recall_score, f1_score, accuracy_score, roc_auc_score
            recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0) if len(y_true)>0 else 0.0
            f1 = f1_score(y_true, y_pred, zero_division=0) if len(y_true)>0 else 0.0
            acc = accuracy_score(y_true, y_pred) if len(y_true)>0 else 0.0
            auc = roc_auc_score(y_true, y_score) if len(set(y_true))>1 else 0.0
        except Exception:
            recall=f1=acc=auc=0.0
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_acc={acc:.4f} val_recall={recall:.4f} f1={f1:.4f} auc={auc:.4f}")
        if recall > best_recall:
            best_recall = recall
            torch.save({"model_state":model.state_dict(),"epoch":epoch},"models/multistream_finetuned_corruptions_best.pt")
            print("Saved best finetune -> models/multistream_finetuned_corruptions_best.pt")
    torch.save({"model_state":model.state_dict(),"epoch":args.epochs},"models/multistream_finetuned_corruptions_last.pt")
    print("Finetune complete. Saved models/multistream_finetuned_corruptions_last.pt")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="models/multistream_best_precomputed_ela.pt")
    parser.add_argument("--rgb-root", type=str, default="data/raw")
    parser.add_argument("--ela-root", type=str, default="data/processed/ELA")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ela-as-rgb", action="store_true")
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--corrupt-prob", type=float, default=0.5)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()
    train(args)
