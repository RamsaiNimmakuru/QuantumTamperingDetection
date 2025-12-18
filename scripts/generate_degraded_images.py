# scripts/generate_degraded_images.py
import os, random, sys
from PIL import Image, ImageFilter
import numpy as np

def degrade_image(img, mode='mixed'):
    if mode == 'jpeg':
        from io import BytesIO
        buf = BytesIO()
        q = random.randint(20, 50)
        img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        return Image.open(buf).convert('RGB')
    if mode == 'noise':
        arr = np.array(img).astype('float32')
        noise = np.random.normal(0, 15, arr.shape).astype('float32')
        arr = (arr + noise).clip(0, 255).astype('uint8')
        return Image.fromarray(arr)
    if mode == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.5)))
    if mode == 'downsample':
        w, h = img.size
        small = img.resize((w//2, h//2), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    # mixed: pick randomly
    choice = random.choice(['jpeg', 'noise', 'blur', 'downsample'])
    return degrade_image(img, choice)

def make_pairs(src_dir, out_clean, out_deg, resize_to=None, mode='mixed'):
    os.makedirs(out_clean, exist_ok=True)
    os.makedirs(out_deg, exist_ok=True)
    files = [f for f in sorted(os.listdir(src_dir)) if f.lower().endswith(('.jpg','.jpeg','.png','.tif','.bmp'))]
    count = 0
    for fn in files:
        src = os.path.join(src_dir, fn)
        try:
            img = Image.open(src).convert('RGB')
        except Exception as e:
            print("Skipping file (can't open):", src, e)
            continue
        if resize_to:
            img = img.resize((resize_to, resize_to))
        # save original as clean
        clean_path = os.path.join(out_clean, fn)
        img.save(clean_path)
        # create degraded
        deg = degrade_image(img, mode)
        deg_path = os.path.join(out_deg, fn)
        deg.save(deg_path)
        count += 1
    print("Created pairs:", count)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python scripts/generate_degraded_images.py <src_dir> <out_clean_dir> <out_deg_dir> [resize] [mode]")
        sys.exit(1)
    src = sys.argv[1]
    out_clean = sys.argv[2]
    out_deg = sys.argv[3]
    resize = int(sys.argv[4]) if len(sys.argv) > 4 else None
    mode = sys.argv[5] if len(sys.argv) > 5 else 'mixed'
    make_pairs(src, out_clean, out_deg, resize_to=resize, mode=mode)
