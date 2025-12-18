# scripts/stress_test.py
"""
Stress test the trained multi-stream model across a degradation matrix.
Usage:
python3 scripts/stress_test.py --pairs-root data/processed/ELA --rgb-root data/raw --ckpt models/multistream_best_precomputed_ela.pt --out results/stress_test.csv
"""

import os
import io      # <-- FIXED: import io
import argparse
import json
import random
from glob import glob
from collections import defaultdict
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import csv
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from src.model_multistream import MultiStreamFusion

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')


def is_image(f):
    return f.lower().endswith(IMG_EXTS)


def build_pairs_simple(rgb_root, ela_root):
    rgb_files = []
    for r, _, fn in os.walk(rgb_root):
        for f in fn:
            if is_image(f):
                rgb_files.append(os.path.join(r, f))

    idx = {os.path.splitext(os.path.basename(p))[0]: p for p in rgb_files}
    pairs = []

    for r, _, fn in os.walk(ela_root):
        for f in fn:
            if not is_image(f):
                continue
            full = os.path.join(r, f)
            base = os.path.splitext(os.path.basename(full))[0]

            # auth=0, tamper=1
            label = 0 if any(k in full.lower() for k in ['auth', 'real', 'orig']) else 1

            rgb = idx.get(base)
            if rgb is None:
                # simple prefix-strip fallback
                s = base
                for prefix in ['au_', 'tp_', 'au', 'tp', 'img_', 'img']:
                    if s.startswith(prefix):
                        s2 = s[len(prefix):]
                        rgb = idx.get(s2)
                        if rgb:
                            break
            if rgb:
                pairs.append((rgb, full, label))
    return pairs


# ---- Degradation functions ----
def degrade_jpeg(img, q=70):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade_downup(img, scale=0.5):
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def degrade_blur(img, radius=1.5):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def degrade_brightness_contrast(img, b=1.0, c=1.0):
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    return img


# ---- Model loading ----
def load_model(ckpt_path, device):
    model = MultiStreamFusion(pretrained_rgb=False)
    ck = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ck, dict) and "model_state" in ck:
        model.load_state_dict(ck["model_state"])
    else:
        model.load_state_dict(ck)

    model = model.to(device)
    model.eval()
    return model


# ---- Preprocessing ----
def preprocess_pair(rgb_img, ela_img, base_size=256):
    # resize both streams
    rgb = rgb_img.resize((base_size, base_size), Image.BILINEAR)
    ela = ela_img.resize((base_size, base_size), Image.BILINEAR)

    # center crop 224
    crop = 224
    x0 = (base_size - crop) // 2
    y0 = (base_size - crop) // 2
    rgb = rgb.crop((x0, y0, x0 + crop, y0 + crop))
    ela = ela.crop((x0, y0, x0 + crop, y0 + crop))

    # convert to CHW normalized
    def pil_to_chw(img):
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]
        return (arr - mean) / std

    rgb_t = torch.from_numpy(pil_to_chw(rgb)).unsqueeze(0).float()
    ela_t = torch.from_numpy(pil_to_chw(ela)).unsqueeze(0).float()

    # residual stream
    from scipy.ndimage import gaussian_filter

    rgb_np = np.asarray(rgb).astype(np.float32)
    blurred = gaussian_filter(rgb_np, sigma=(1, 1, 0))
    residual = np.clip((rgb_np - blurred) / 255.0, -1.0, 1.0)
    residual = residual.transpose(2, 0, 1)
    res_t = torch.from_numpy(residual).unsqueeze(0).float()

    return rgb_t, ela_t, res_t


# ---- Main test ----
def run_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    pairs = build_pairs_simple(args.rgb_root, args.ela_root)
    print("Pairs discovered:", len(pairs))

    degradations = [
        ("clean", lambda x: x),
        ("jpeg_q_90", lambda x: degrade_jpeg(x, 90)),
        ("jpeg_q_70", lambda x: degrade_jpeg(x, 70)),
        ("jpeg_q_40", lambda x: degrade_jpeg(x, 40)),
        ("down_0.5", lambda x: degrade_downup(x, 0.5)),
        ("down_0.25", lambda x: degrade_downup(x, 0.25)),
        ("blur_r1", lambda x: degrade_blur(x, 1.0)),
        ("blur_r2", lambda x: degrade_blur(x, 2.0)),
        ("bright_0.8", lambda x: degrade_brightness_contrast(x, 0.8, 1.0)),
        ("contrast_0.8", lambda x: degrade_brightness_contrast(x, 1.0, 0.8)),
    ]

    # sample subset
    if args.max_samples is None:
        chosen = pairs
    else:
        chosen = random.sample(pairs, min(args.max_samples, len(pairs)))

    rows = []
    os.makedirs(args.sample_dir, exist_ok=True)

    for name, func in degradations:
        y_true = []
        y_pred = []
        y_score = []

        pbar = tqdm(chosen, desc=name)
        for rgb_p, ela_p, label in pbar:
            try:
                rgb = Image.open(rgb_p).convert("RGB")
                ela_gray = Image.open(ela_p).convert("L")
                ela_rgb = Image.merge("RGB", (ela_gray, ela_gray, ela_gray))
            except:
                continue

            rgb_deg = func(rgb)

            rgb_t, ela_t, res_t = preprocess_pair(rgb_deg, ela_rgb, args.base_size)
            rgb_t = rgb_t.to(device)
            ela_t = ela_t.to(device)
            res_t = res_t.to(device)

            with torch.no_grad():
                out = model(rgb_t, ela_t, res_t)
                prob = F.softmax(out, dim=1)[:, 1].item()

            y_score.append(prob)
            y_pred.append(1 if prob >= 0.5 else 0)
            y_true.append(label)

        # metrics
        if len(set(y_true)) > 1:
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred, pos_label=1)
            try:
                auc = roc_auc_score(y_true, y_score)
            except:
                auc = float("nan")
        else:
            acc = f1 = rec = auc = float("nan")

        rows.append(
            {
                "degradation": name,
                "n": len(y_true),
                "acc": acc,
                "f1": f1,
                "recall_tampered": rec,
                "auc": auc,
            }
        )

    # write CSV
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["degradation", "n", "acc", "f1", "recall_tampered", "auc"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("Stress test complete. Results saved to:", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-root", type=str, default="data/raw")
    parser.add_argument("--ela-root", type=str, default="data/processed/ELA")
    parser.add_argument("--ckpt", type=str, default="models/multistream_best_precomputed_ela.pt")
    parser.add_argument("--out", type=str, default="results/stress_test.csv")
    parser.add_argument("--sample-dir", type=str, default="results/stress_samples")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--base-size", type=int, default=256)
    args = parser.parse_args()
    run_test(args)
