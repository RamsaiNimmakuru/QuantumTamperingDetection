# scripts/regenerate_columbia_ela.py
import os, io, shutil
from PIL import Image, ImageChops
from tqdm import tqdm
from pathlib import Path

SRC_AUTH = "data/raw/Columbia/4cam_auth/4cam_auth"
SRC_TAMP = "data/raw/Columbia/4cam_splc/4cam_splc"
OUT = "data/processed/ELA_columbia"

JPEG_QUALITY = 95   # MATCH training
RESIZE = (224, 224) # MATCH training

def generate_ela(img_path):
    img = Image.open(img_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=JPEG_QUALITY)
    comp = Image.open(buf).convert("RGB")
    ela = ImageChops.difference(img, comp)
    # scale ELA for visibility
    extrema = ela.getextrema()
    # compute scale avoiding zeros
    scales = []
    for ch in extrema:
        if ch[1] != 0:
            scales.append(255.0 / ch[1])
    scale = max(scales) if scales else 1.0
    ela = Image.eval(ela, lambda x: int(min(255, x * scale)))
    return ela.resize(RESIZE)

def process(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff"))]
    for f in tqdm(files, desc=f"Processing {src_dir}"):
        try:
            ela = generate_ela(os.path.join(src_dir, f))
            ela.save(os.path.join(out_dir, Path(f).stem + ".jpg"), quality=95)
        except Exception as e:
            print("SKIP", f, e)

if __name__ == "__main__":
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "authentic"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "tampered"), exist_ok=True)
    process(SRC_AUTH, os.path.join(OUT, "authentic"))
    process(SRC_TAMP, os.path.join(OUT, "tampered"))
    print("✅ Regenerated Columbia ELA under", OUT)
