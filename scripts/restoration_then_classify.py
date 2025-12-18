# scripts/restoration_then_classify.py
"""
Pipeline:
  - For each image in a folder (src), optionally restore (if restored_folder provided) else use src
  - Run multistream classifier on (rgb, ela, residual)
  - Save per-image predictions and compute overall metrics (acc, f1, recall, auc)
Usage:
python3 scripts/restoration_then_classify.py \
  --ckpt models/multistream_best_precomputed_ela.pt \
  --rgb-root data/pairs_bsd500/val/degraded \
  --ela-root data/pairs_bsd500/val/ela_or_processed_if_any \
  --restored results/restored_bsd500 \
  --out results/classify_restored_metrics.json
"""
import os, json, argparse
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from src.model_multistream import MultiStreamFusion

def load_image(path):
    img = Image.open(path).convert('RGB')
    return img

def prepare_tensors(rgb_img, ela_img, base_size=256):
    # resize center crop to 224 same as other scripts
    rgb = rgb_img.resize((base_size, base_size), Image.BILINEAR)
    ela = ela_img.resize((base_size, base_size), Image.BILINEAR)
    x0 = (base_size - 224)//2
    y0 = x0
    rgb = rgb.crop((x0,y0,x0+224,y0+224))
    ela = ela.crop((x0,y0,x0+224,y0+224))
    # normalize
    arr_rgb = np.asarray(rgb).astype('float32')/255.0
    mean = np.array([0.485,0.456,0.406])[None,None,:]
    std = np.array([0.229,0.224,0.225])[None,None,:]
    arr_rgb = (arr_rgb - mean)/std
    arr_rgb = arr_rgb.transpose(2,0,1)
    rgb_t = torch.from_numpy(arr_rgb).unsqueeze(0).float()
    arr_ela = np.asarray(ela).astype('float32')/255.0
    arr_ela = (arr_ela - mean)/std
    arr_ela = arr_ela.transpose(2,0,1)
    ela_t = torch.from_numpy(arr_ela).unsqueeze(0).float()
    # residual
    from scipy.ndimage import gaussian_filter
    rgb_np = np.asarray(rgb).astype('float32')
    blurred = gaussian_filter(rgb_np, sigma=(1,1,0))
    residual = np.clip((rgb_np - blurred)/255.0, -1.0, 1.0)
    residual_t = torch.from_numpy(residual.transpose(2,0,1)).unsqueeze(0).float()
    return rgb_t, ela_t, residual_t

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiStreamFusion(pretrained_rgb=False)
    ck = torch.load(args.ckpt, map_location='cpu')
    if 'model_state' in ck: model.load_state_dict(ck['model_state'])
    else: model.load_state_dict(ck)
    model.to(device).eval()
    # build list of pairs from rgb-root (expected filename base) and labels from ela-root subfolders or from naming
    rgb_files = []
    for r,_,fn in os.walk(args.rgb_root):
        for f in fn:
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                rgb_files.append(os.path.join(r,f))
    rgb_files.sort()
    y_true=[]; y_pred=[]; y_score=[]
    per_image = []
    for p in tqdm(rgb_files):
        base = os.path.splitext(os.path.basename(p))[0]
        # determine label: try to find in ela-root path if naming contains 'clean' or 'degraded' or parent folder
        # fallback: check whether parent folder name contains 'clean' or 'tampered'
        parent = os.path.basename(os.path.dirname(p)).lower()
        if 'clean' in parent or 'au' in parent or 'auth' in parent or 'real' in parent:
            label = 0
        else:
            label = 1
        # choose image to classify: restored if available else original
        img_inp = p
        if args.restored:
            cand = os.path.join(args.restored, os.path.basename(p))
            if os.path.exists(cand): img_inp = cand
        # ELA image: prefer ela-root with same basename else generate quick ELA from rgb
        ela_path = None
        if args.ela_root:
            candidate = os.path.join(args.ela_root, os.path.basename(p))
            if os.path.exists(candidate): ela_path = candidate
        if ela_path is None:
            # create a grayscale copy as ELA surrogate
            ela_img = Image.open(p).convert('L')
            ela_img = Image.merge('RGB', (ela_img, ela_img, ela_img))
        else:
            ela_img = Image.open(ela_path).convert('RGB')
        rgb_img = Image.open(img_inp).convert('RGB')
        rgb_t, ela_t, res_t = prepare_tensors(rgb_img, ela_img, base_size=args.base_size)
        rgb_t = rgb_t.to(device); ela_t = ela_t.to(device); res_t = res_t.to(device)
        with torch.no_grad():
            out = model(rgb_t, ela_t, res_t)
            prob = torch.softmax(out, dim=1)[0,1].item()
            pred = 1 if prob>=0.5 else 0
        y_true.append(label); y_pred.append(pred); y_score.append(prob)
        per_image.append({'file':os.path.basename(p),'label':int(label),'pred':int(pred),'score':float(prob)})
    # metrics
    metrics = {}
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['recall_tampered'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['auc'] = roc_auc_score(y_true, y_score) if len(set(y_true))>1 else None
    except Exception as e:
        metrics['error'] = str(e)
    # save
    out = {'metrics':metrics,'per_image':per_image}
    with open(args.out,'w') as f:
        json.dump(out,f,indent=2)
    print("Saved results to", args.out)
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--rgb-root", required=True)
    p.add_argument("--ela-root", default=None)
    p.add_argument("--restored", default=None)
    p.add_argument("--out", default="results/classify_restored_metrics.json")
    p.add_argument("--base-size", type=int, default=256)
    args = p.parse_args()
    main(args)
