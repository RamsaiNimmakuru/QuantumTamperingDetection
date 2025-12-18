# scripts/train_multistream_precomputed.py
"""
Training script: pairs RAW RGB images + precomputed ELA images, trains MultiStreamFusion.
- Robust pairing heuristics
- Safe image open and resizing to avoid shape mismatch
- Augmentations compatible with 6-channel joint input
- Checkpointing by tampered recall
"""
import os
import random
import argparse
import json
from glob import glob
from collections import defaultdict
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Albumentations imports (use Resize + RandomCrop and RandomBrightnessContrast)
from albumentations import (
    Compose,
    Resize,
    RandomCrop,
    HorizontalFlip,
    RandomBrightnessContrast,
    Normalize,
)
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score
from tqdm import tqdm

from src.model_multistream import MultiStreamFusion

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')


def is_image(fname):
    return fname.lower().endswith(IMG_EXTS)


def find_all_images(root):
    root = os.path.expanduser(root)
    files = []
    for r, _, fnames in os.walk(root):
        for f in fnames:
            if is_image(f):
                files.append(os.path.join(r, f))
    return files


def basename_no_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def build_pairs(rgb_root, ela_root, verbose=True):
    """
    Build pairs list: for each ela file under ela_root, find a matching rgb file under rgb_root.
    Returns:
      pairs: list of (rgb_path, ela_path, label) where label: 0=Real/authentic, 1=Tampered
    """
    ela_files = []
    for root, _, fnames in os.walk(ela_root):
        for f in fnames:
            if is_image(f):
                full = os.path.join(root, f)
                lower = full.lower()
                if any(k in lower for k in ['/auth', '/real', '/orig', '/genuine', '/authentic']):
                    label = 0
                elif any(k in lower for k in ['/tamper', '/tampered', '/fake', '/forged', '/splice', '/tp', '/splc']):
                    label = 1
                else:
                    parent = os.path.basename(os.path.dirname(full)).lower()
                    if 'auth' in parent or 'real' in parent or 'orig' in parent:
                        label = 0
                    elif 'tamper' in parent or 'tampered' in parent or 'fake' in parent or 'tp' in parent or 'splc' in parent:
                        label = 1
                    else:
                        label = None
                if label is not None:
                    ela_files.append((full, label))

    if verbose:
        print(f"Found {len(ela_files)} ELA images under {ela_root} (with label assignment attempts)")

    rgb_files = find_all_images(rgb_root)
    if verbose:
        print(f"Discovered {len(rgb_files)} candidate RGB images under {rgb_root}")

    idx_by_basename = defaultdict(list)
    for p in rgb_files:
        idx_by_basename[basename_no_ext(p)].append(p)

    pairs = []
    skipped = 0
    for ela_path, label in ela_files:
        ela_base = basename_no_ext(ela_path)
        found_rgb = None

        # heuristic 1: exact basename match
        if ela_base in idx_by_basename:
            found_rgb = idx_by_basename[ela_base][0]
        else:
            # heuristic 2: substring match
            for k, lst in idx_by_basename.items():
                if k in ela_base or ela_base in k:
                    found_rgb = lst[0]
                    break

        if not found_rgb:
            # heuristic 3: strip common prefixes
            short = ela_base
            for prefix in ['au_', 'tp_', 'au', 'tp', 'img_', 'img', 'pla', 'nat', 'sec', 'arc', 'ind', 'txt']:
                if short.startswith(prefix):
                    cand = short[len(prefix):]
                    if cand in idx_by_basename:
                        found_rgb = idx_by_basename[cand][0]
                        break

        if not found_rgb:
            # heuristic 4: slow fallback substring search
            for p in rgb_files:
                b = basename_no_ext(p)
                if ela_base in b or b in ela_base:
                    found_rgb = p
                    break

        if found_rgb:
            pairs.append((found_rgb, ela_path, label))
        else:
            skipped += 1
            if verbose and skipped <= 20:
                print(f"Warning: no RGB match found for ELA {ela_path} (skipping).")
    if verbose:
        print(f"Paired {len(pairs)} ELA images with RGB images; skipped {skipped} unmatched ELA files.")
        c0 = sum(1 for _, _, l in pairs if l == 0)
        c1 = sum(1 for _, _, l in pairs if l == 1)
        print(f"Class counts -> Real: {c0}, Tampered: {c1}")
    return pairs


# ---- Robust Paired Dataset ----
class PairedRGBELADataset(Dataset):
    def __init__(self, pairs, transform=None, ela_as_rgb=False, base_size=256, max_open_attempts=3):
        """
        pairs: list of (rgb_path, ela_path, label)
        transform: albumentations transform applied jointly to rgb and ela concatenated
        ela_as_rgb: if True assumes ela image is 3-channel RGB image (common if precomputed), else it's treated as grayscale and stacked to 3-ch by repeat
        base_size: size to which both images will be resized before forming the joint array (keeps shapes consistent)
        """
        self.pairs = pairs
        self.transform = transform
        self.ela_as_rgb = ela_as_rgb
        self.base_size = int(base_size)
        self.max_open_attempts = int(max_open_attempts)
        self._bad_examples_logged = 0

    def __len__(self):
        return len(self.pairs)

    def _open_image_safe(self, path, mode='RGB'):
        # attempt to open image safely multiple times (handles transient IO)
        last_err = None
        for _ in range(self.max_open_attempts):
            try:
                img = Image.open(path)
                if mode == 'RGB':
                    img = img.convert('RGB')
                elif mode == 'L':
                    img = img.convert('L')
                return img
            except Exception as e:
                last_err = e
        # raise last error if all attempts fail
        raise last_err

    def __getitem__(self, idx):
        rgb_p, ela_p, label = self.pairs[idx]
        try:
            rgb = self._open_image_safe(rgb_p, mode='RGB')
            ela = self._open_image_safe(ela_p, mode='RGB') if self.ela_as_rgb else self._open_image_safe(ela_p, mode='L')
        except Exception as e:
            # if an image fails catastrophically, log and return a random other sample (robust)
            if self._bad_examples_logged < 10:
                print(f"Warning: failed to open image(s) rgb={rgb_p} ela={ela_p} -> {e}")
                self._bad_examples_logged += 1
            alt_idx = random.randint(0, len(self.pairs) - 1)
            return self.__getitem__(alt_idx)

        # Resize both images to base_size x base_size to ensure matching shapes
        rgb = rgb.resize((self.base_size, self.base_size), resample=Image.BILINEAR)

        if ela.mode == 'L':
            # convert L->RGB and resize
            ela_rgb = Image.merge('RGB', (ela, ela, ela))
            ela_rgb = ela_rgb.resize((self.base_size, self.base_size), resample=Image.BILINEAR)
        else:
            ela_rgb = ela.convert('RGB')
            ela_rgb = ela_rgb.resize((self.base_size, self.base_size), resample=Image.BILINEAR)

        rgb_np = np.array(rgb)
        ela_np = np.array(ela_rgb)

        # Defensive check — sizes must match now
        if rgb_np.shape[0] != ela_np.shape[0] or rgb_np.shape[1] != ela_np.shape[1]:
            if self._bad_examples_logged < 20:
                print(f"Mismatch after resize: rgb={rgb_p} ({rgb_np.shape}) ela={ela_p} ({ela_np.shape})")
                self._bad_examples_logged += 1
            # Force resize ela to rgb shape
            ela_rgb = Image.fromarray(ela_np).resize((rgb_np.shape[1], rgb_np.shape[0]), resample=Image.BILINEAR)
            ela_np = np.array(ela_rgb)

        # Concatenate into H x W x 6
        joint = np.concatenate([rgb_np, ela_np], axis=2)  # H W 6

        if self.transform is not None:
            aug = self.transform(image=joint)
            aug_joint = aug['image']
            # albumentations + ToTensorV2 gives torch.Tensor CHW on 'image'
            rgb_t = aug_joint[:3, ...].float()
            ela_t = aug_joint[3:6, ...].float()
        else:
            toT = T.ToTensor()
            rgb_t = toT(Image.fromarray(rgb_np))
            ela_t = toT(Image.fromarray(ela_np))

        # residual: rgb - gaussian_blur(rgb)
        # rgb_t is tensor CHW in range [0,1], bring to HWC float [0..255] for blur
        rgb_np_f = np.transpose(rgb_t.numpy(), (1, 2, 0)) * 255.0
        try:
            from scipy.ndimage import gaussian_filter
            blurred = gaussian_filter(rgb_np_f, sigma=(1, 1, 0))
        except Exception:
            blurred = rgb_np_f
        residual = rgb_np_f - blurred
        residual = np.clip(residual / 255.0, -1.0, 1.0)
        residual_t = torch.from_numpy(np.transpose(residual, (2, 0, 1))).float()

        return rgb_t, ela_t, residual_t, torch.tensor(label, dtype=torch.long)
# ---- end PairedRGBELADataset ----


def compute_metrics(y_true, y_pred, y_score):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)
    acc = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else 0.0
    auc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else 0.0
    rec_tampered = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)) if len(y_true) > 0 else 0.0
    return dict(acc=acc, f1=f1, auc=auc, recall_tampered=rec_tampered)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    pairs = build_pairs(args.rgb_root, args.ela_root, verbose=True)
    if len(pairs) == 0:
        raise RuntimeError("No pairs found. Check rgb-root and ela-root paths or pairing heuristics.")

    random.seed(42)
    random.shuffle(pairs)
    n = len(pairs)
    ntrain = int(0.8 * n)
    train_pairs = pairs[:ntrain]
    val_pairs = pairs[ntrain:]

    # Use Resize -> RandomCrop for compatibility
    train_tf = Compose(
        [
            Resize(height=256, width=256, p=1.0),
            RandomCrop(height=224, width=224, p=1.0),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    val_tf = Compose(
        [
            Resize(height=224, width=224, p=1.0),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    train_ds = PairedRGBELADataset(train_pairs, transform=train_tf, ela_as_rgb=args.ela_as_rgb, base_size=args.base_size)
    val_ds = PairedRGBELADataset(val_pairs, transform=val_tf, ela_as_rgb=args.ela_as_rgb, base_size=args.base_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = MultiStreamFusion(pretrained_rgb=True).to(device)

    # freeze rgb backbone initially
    for p in model.rgb.features.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_recall = 0.0
    best_metrics = None
    y_true_all = []
    y_pred_all = []
    y_score_all = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for rgb, ela, res, label in pbar:
            rgb = rgb.to(device)
            ela = ela.to(device)
            res = res.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(rgb, ela, res)
                loss = criterion(out, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * rgb.size(0)
        train_loss = running_loss / len(train_ds)

        # validation
        model.eval()
        y_true = []
        y_pred = []
        y_score = []
        val_loss = 0.0
        with torch.no_grad():
            for rgb, ela, res, label in val_loader:
                rgb = rgb.to(device)
                ela = ela.to(device)
                res = res.to(device)
                label = label.to(device)
                out = model(rgb, ela, res)
                loss = criterion(out, label)
                val_loss += float(loss.item()) * rgb.size(0)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                y_true.extend(label.cpu().numpy().tolist())
                y_pred.extend(preds.tolist())
                y_score.extend(probs.tolist())
        val_loss = val_loss / len(val_ds)
        metrics = compute_metrics(y_true, y_pred, y_score)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={metrics['acc']:.4f} "
            f"val_recall={metrics['recall_tampered']:.4f} f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}"
        )

        # save last epoch preds for report
        y_true_all = y_true
        y_pred_all = y_pred
        y_score_all = y_score

        # checkpoint by tampered recall
        if metrics["recall_tampered"] > best_recall:
            best_recall = metrics["recall_tampered"]
            best_metrics = metrics
            ckpt = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "metrics": metrics}
            torch.save(ckpt, f"models/multistream_best_precomputed_ela.pt")
            print("Saved best -> models/multistream_best_precomputed_ela.pt")

        # unfreeze if requested
        if epoch == args.unfreeze_epoch:
            print("Unfreezing RGB backbone layers")
            for name, param in model.rgb.features.named_parameters():
                param.requires_grad = True

    # final save & reports
    torch.save({"model_state": model.state_dict()}, "models/multistream_last_precomputed_ela.pt")
    import numpy as _np

    _np.save("results/y_true.npy", _np.array(y_true_all))
    _np.save("results/y_pred.npy", _np.array(y_pred_all))
    _np.save("results/y_scores.npy", _np.array(y_score_all))
    with open("results/classification_report.txt", "w") as f:
        f.write(classification_report(y_true_all, y_pred_all, target_names=["Real", "Tampered"]))
    with open("results/metrics_summary.json", "w") as f:
        json.dump(best_metrics if best_metrics is not None else metrics, f, indent=2)
    print("Training complete. Artifacts in models/ and results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-root", type=str, default="data/raw", help="root containing RGB images (recursively searched)")
    parser.add_argument("--ela-root", type=str, default="data/processed/ELA", help="root containing precomputed ELA images (authentic/tampered subfolders)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze-epoch", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ela-as-rgb", action="store_true", help="set if ELA files are precomputed and already 3-channel RGB images")
    parser.add_argument("--base-size", type=int, default=256, help="base resize size for both RGB and ELA before augmentation")
    args = parser.parse_args()
    train(args)
