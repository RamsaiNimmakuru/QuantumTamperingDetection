import os, shutil, random
from tqdm import tqdm
from PIL import Image

CASIA_DIR = "data/raw/CASIA"
COLUMBIA_DIR = "data/raw/Columbia"
OUT_IMG = "data/processed/localization/images"
OUT_MASK = "data/processed/localization/masks"
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)

def is_valid(path):
    try:
        Image.open(path).verify()
        return True
    except:
        return False

def copy_casia():
    src_tp = os.path.join(CASIA_DIR, "Tp")
    src_gt = os.path.join(CASIA_DIR, "CASIA 2 Groundtruth")
    count = 0
    for f in tqdm(os.listdir(src_tp), desc="CASIA"):
        base = os.path.splitext(f)[0]
        img_path = os.path.join(src_tp, f)
        for mask_ext in [".png", "_gt.png", ".bmp"]:
            mask_path = os.path.join(src_gt, base + mask_ext)
            if os.path.exists(mask_path) and is_valid(img_path) and is_valid(mask_path):
                img = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")
                img.save(os.path.join(OUT_IMG, f"casia_{base}.png"))
                mask.save(os.path.join(OUT_MASK, f"casia_{base}.png"))
                count += 1
                break
    print(f"✅ Copied {count} CASIA pairs")

def copy_columbia():
    src_img = os.path.join(COLUMBIA_DIR, "4cam_splc", "4cam_splc")
    src_mask = os.path.join(COLUMBIA_DIR, "4cam_splc", "edgemask")
    count = 0
    for f in tqdm(os.listdir(src_mask), desc="Columbia"):
        base = os.path.splitext(f)[0]
        for ext in [".tif", ".png", ".jpg"]:
            img_path = os.path.join(src_img, base + ext)
            if os.path.exists(img_path) and is_valid(img_path):
                mask_path = os.path.join(src_mask, f)
                img = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")
                img.save(os.path.join(OUT_IMG, f"col_{base}.png"))
                mask.save(os.path.join(OUT_MASK, f"col_{base}.png"))
                count += 1
                break
    print(f"✅ Copied {count} Columbia pairs")

def split_data():
    imgs = sorted(os.listdir(OUT_IMG))
    random.shuffle(imgs)
    n = len(imgs)
    tr = imgs[:int(0.8*n)]
    va = imgs[int(0.8*n):]
    with open("data/processed/localization/train.txt","w") as f:
        for i in tr: f.write(f"data/processed/localization/images/{i} data/processed/localization/masks/{i}\n")
    with open("data/processed/localization/val.txt","w") as f:
        for i in va: f.write(f"data/processed/localization/images/{i} data/processed/localization/masks/{i}\n")
    print(f"Train:{len(tr)} Val:{len(va)}")

if __name__=="__main__":
    copy_casia()
    copy_columbia()
    split_data()
    print("✅ Localization dataset ready (CASIA + Columbia)")
