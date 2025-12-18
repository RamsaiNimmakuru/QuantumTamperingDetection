# scripts/evaluate_restoration.py
import os, json, argparse, math
from PIL import Image
import numpy as np
from pathlib import Path

def psnr_img(a, b):
    a = np.array(a).astype('float32')/255.0
    b = np.array(b).astype('float32')/255.0
    mse = ((a - b) ** 2).mean()
    if mse == 0:
        return float('inf')
    return float(20 * math.log10(1.0 / math.sqrt(mse)))

def evaluate_pairs(clean_dir, restored_dir):
    clean_map = {p.stem: p for p in Path(clean_dir).glob("*.*")}
    res_map = {p.stem: p for p in Path(restored_dir).glob("*.*")}
    psnrs = []
    matched = 0
    for stem, cpath in clean_map.items():
        if stem in res_map:
            rpath = res_map[stem]
            c = Image.open(cpath).convert('RGB').resize((256,256))
            r = Image.open(rpath).convert('RGB').resize((256,256))
            psnrs.append(psnr_img(c, r))
            matched += 1
    avg = float(np.mean(psnrs)) if len(psnrs) > 0 else 0.0
    return {"matched": int(matched), "avg_psnr": float(avg)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True)
    parser.add_argument("--restored", required=True)
    parser.add_argument("--out", default="results/restoration_eval.json")
    args = parser.parse_args()
    res = evaluate_pairs(args.clean, args.restored)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("Saved", args.out, res)
