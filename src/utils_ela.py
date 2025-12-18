'''
import os
from PIL import Image, ImageChops, ImageEnhance
from tqdm import tqdm

def generate_ela_image(path, quality=90):
    orig = Image.open(path).convert('RGB')
    tmp = 'temp_ela.jpg'
    orig.save(tmp, 'JPEG', quality=quality)
    resaved = Image.open(tmp)
    ela = ImageChops.difference(orig, resaved)
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff else 1
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    return ela

def batch_generate_ela(input_dir, output_dir, mapping=None, quality=90):
    if mapping is None:
        mapping = {"Au": "authentic", "Tp": "tampered"}
    os.makedirs(output_dir, exist_ok=True)

    for sub in mapping:
        src = os.path.join(input_dir, sub)
        dst = os.path.join(output_dir, mapping[sub])
        if not os.path.exists(src):
            print(f"⚠️ Skipping {src} (not found)")
            continue
        os.makedirs(dst, exist_ok=True)
        print(f"📁 Processing {src} → {dst}")

        for f in tqdm(os.listdir(src)):
            if not f.lower().endswith(('jpg', 'jpeg', 'png')):
                continue
            try:
                ela = generate_ela_image(os.path.join(src, f), quality)
                ela.save(os.path.join(dst, os.path.splitext(f)[0] + '.png'))
            except Exception as e:
                print(f"❌ Error: {f} → {e}")
'''

from PIL import Image, ImageChops
import numpy as np
import io

def generate_ela(image_path, quality=95):
    img = Image.open(image_path).convert("RGB")

    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality)
    img_compressed = Image.open(buffer)

    ela = ImageChops.difference(img, img_compressed)
    extrema = ela.getextrema()

    scale = 1
    for channel in extrema:
        if channel[1] != 0:
            scale = max(scale, 255 // channel[1])

    ela = Image.eval(ela, lambda x: x * scale)
    ela = ela.resize((224, 224))

    return np.array(ela)
