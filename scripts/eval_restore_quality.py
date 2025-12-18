#!/usr/bin/env python3
# scripts/eval_restored_quality.py
import argparse, os, glob, csv
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from pathlib import Path

def load_img(path, size=None):
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize(size, Image.BILINEAR)
    return np.asarray(im)

def compute_metrics(gt_np, pred_np):
    # gt and pred are H,W,3 uint8
    if gt_np.shape != pred_np.shape:
        # resize pred to gt
        from PIL import Image
        pred_np = np.asarray(Image.fromarray(pred_np).resize((gt_np.shape[1], gt_np.shape[0]), Image.BILINEAR))
    # convert to float in [0,1]
    gt_f = gt_np.astype(np.float32) / 255.0
    pred_f = pred_np.astype(np.float32) / 255.0
    # compute PSNR (skimage expects range 1.0)
    p = psnr(gt_f, pred_f, data_range=1.0)
    s = ssim(gt_f, pred_f, data_range=1.0, multichannel=True)
    return p, s

def main(args):
    restored = Path(args.restored_folder)
    gt = Path(args.gt_folder) if args.gt_folder else None
    out_csv = Path(args.out_csv)
    out_html = Path(args.out_html)
    restored_files = sorted([p for p in restored.glob("*") if p.suffix.lower() in (".jpg",".png",".jpeg",".bmp")])
    rows = []
    html_lines = ['<html><body><h1>Restoration preview</h1><table border="1">']
    html_lines.append("<tr><th>file</th><th>ground-truth</th><th>restored</th><th>PSNR</th><th>SSIM</th></tr>")
    for rf in restored_files:
        name = rf.name
        gt_path = None
        if gt:
            # try exact filename match, then try name with prefix/suffix permutations
            cand = gt / name
            if cand.exists():
                gt_path = cand
            else:
                # try removing prefixes like "orig_" or "restored_"
                for prefix in ("orig_", "orig-", "gt_", "gt-"):
                    cand2 = gt / (name.replace(prefix, ""))
                    if cand2.exists():
                        gt_path = cand2
                        break
        if not gt_path:
            # skip metrics if no gt
            rows.append([name, "", str(rf), "", ""])
            html_lines.append(f"<tr><td>{name}</td><td>—</td><td><img src='{rf.as_posix()}' width=256></td><td>—</td><td>—</td></tr>")
            continue
        try:
            gt_np = load_img(gt_path)
            pred_np = load_img(rf)
            p, s = compute_metrics(gt_np, pred_np)
            rows.append([name, str(gt_path), str(rf), f"{p:.4f}", f"{s:.4f}"])
            html_lines.append(f"<tr><td>{name}</td><td><img src='{gt_path.as_posix()}' width=256></td><td><img src='{rf.as_posix()}' width=256></td><td>{p:.3f}</td><td>{s:.3f}</td></tr>")
        except Exception as e:
            rows.append([name, str(gt_path) if gt_path else "", str(rf), "err", "err"])
            html_lines.append(f"<tr><td>{name}</td><td>{gt_path}</td><td>{rf.as_posix()}</td><td>err</td><td>err</td></tr>")

    html_lines.append("</table></body></html>")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "gt_path", "restored_path", "psnr", "ssim"])
        w.writerows(rows)
    with open(out_html, "w") as f:
        f.write("\n".join(html_lines))
    print("Saved CSV ->", out_csv)
    print("Saved preview HTML ->", out_html)
    # print summary stats (if psnr values present)
    vals = [ (float(r[3]), float(r[4])) for r in rows if r[3] not in ("", "err") ]
    if vals:
        ps = [v[0] for v in vals]
        ss = [v[1] for v in vals]
        import statistics
        print("PSNR mean/std/min/max:", statistics.mean(ps), statistics.pstdev(ps) if len(ps)>1 else 0, min(ps), max(ps))
        print("SSIM mean/std/min/max:", statistics.mean(ss), statistics.pstdev(ss) if len(ss)>1 else 0, min(ss), max(ss))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--restored-folder", required=True)
    p.add_argument("--gt-folder", required=False, default=None)
    p.add_argument("--out-csv", required=False, default="results/restoration_quality.csv")
    p.add_argument("--out-html", required=False, default="results/restoration_preview.html")
    args = p.parse_args()
    main(args)
