# scripts/generate_columbia_ela.py
import os
from PIL import Image, ImageChops
import io
import numpy as np
from pathlib import Path
from tqdm import tqdm

SRC_AUTH = "data/raw/Columbia/4cam_auth/4cam_auth"
SRC_TAMP = "data/raw/Columbia/4cam_splc/4cam_splc"
OUT_BASE = "data/processed/ELA_columbia"
SIZE = (224, 224)     # must match model input
JPEG_QUALITY = 95     # must match training ELA quality

def generate_ela_array(img_path, quality=JPEG_QUALITY, size=SIZE):
    img = Image.open(img_path).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality)
    img_compressed = Image.open(buffer).convert("RGB")
    ela = ImageChops.difference(img, img_compressed)
    # scale channels
    extrema = ela.getextrema()
    scale = 1
    for ch in extrema:
        if ch[1] != 0:
            scale = max(scale, 255 // ch[1])
    ela = Image.eval(ela, lambda x: x * scale)
    ela = ela.resize(size)
    return np.array(ela)

def save_ela_set(src_dir, out_dir, split_ratio=0.8):
    files = [f for f in sorted(os.listdir(src_dir)) if f.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff"))]
    n = len(files)
    split = int(n * split_ratio)
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)
    for i, f in enumerate(tqdm(files, desc=f"Processing {src_dir}")):
        src = os.path.join(src_dir, f)
        arr = generate_ela_array(src)
        out_name = f"{Path(f).stem}.jpg"
        if i < split:
            dst = os.path.join(out_dir, "train", out_name)
        else:
            dst = os.path.join(out_dir, "val", out_name)
        Image.fromarray(arr.astype("uint8")).save(dst, quality=95)

def copy_to_test(src_dir, out_dir_test):
    # for completeness, we'll copy all into test folder as well (optional)
    os.makedirs(out_dir_test, exist_ok=True)
    for f in sorted(os.listdir(src_dir)):
        if f.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff")):
            src = os.path.join(src_dir, f)
            arr = generate_ela_array(src)
            Image.fromarray(arr.astype("uint8")).save(os.path.join(out_dir_test, f"{Path(f).stem}.jpg"), quality=95)

if __name__ == "__main__":
    # Authentic
    save_ela_set(SRC_AUTH, os.path.join(OUT_BASE, "authentic"))
    # Tampered
    save_ela_set(SRC_TAMP, os.path.join(OUT_BASE, "tampered"))
    # Optional: create separate test folder with full images (if needed)
    # copy_to_test(SRC_AUTH, os.path.join(OUT_BASE,"test","authentic"))
    # copy_to_test(SRC_TAMP, os.path.join(OUT_BASE,"test","tampered"))
    print("✅ Columbia ELA dataset generated under", OUT_BASE)
