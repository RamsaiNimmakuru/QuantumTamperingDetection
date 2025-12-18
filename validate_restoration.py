# validate_restoration.py
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
import csv
import importlib

# Utilities
def to_tensor(img):
    if isinstance(img, Image.Image):
        return T.ToTensor()(img)  # CxHxW, float 0..1
    if isinstance(img, np.ndarray):
        # HxW[xC]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2,0,1) if img.ndim==3 else torch.from_numpy(img).unsqueeze(0)
    if torch.is_tensor(img):
        return img
    raise TypeError("Unsupported image type: "+str(type(img)))

def tensor_to_pil(t):
    # t: torch tensor CxHxW or 1xHxW
    if torch.is_tensor(t):
        t = t.detach().cpu()
        if t.ndim == 3:
            return T.ToPILImage()(t.clamp(0,1))
        if t.ndim==4 and t.shape[0]==1:
            return T.ToPILImage()(t[0].clamp(0,1))
        if t.ndim==2:
            arr = (t.numpy()*255).astype(np.uint8)
            return Image.fromarray(arr)
    raise TypeError("Unsupported tensor shape for conversion to PIL: "+str(getattr(t,'shape',None)))

def pil_or_tensor_to_pil(x):
    if isinstance(x, Image.Image):
        return x
    if torch.is_tensor(x) or isinstance(x, np.ndarray):
        return tensor_to_pil(to_tensor(x))
    raise TypeError("Unsupported output type: "+str(type(x)))

def safe_load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt

def strip_module_prefix(sd):
    new = {}
    for k,v in sd.items():
        if k.startswith("module."):
            new[k[7:]] = v
        else:
            new[k] = v
    return new

def try_instantiate_known_models():
    """
    Try to import known model classes from repo.
    Return dictionary name->class
    """
    candidates = {}
    # common filenames from your repo
    try:
        m = importlib.import_module("model_unet")
        if hasattr(m, "UNet"):
            candidates["unet.UNet"] = m.UNet
        if hasattr(m, "RestorationUNet"):
            candidates["unet.RestorationUNet"] = m.RestorationUNet
    except Exception:
        pass
    try:
        m = importlib.import_module("models_restoration")
        for nm in dir(m):
            if "Unet" in nm or "UNet" in nm or "Restor" in nm:
                candidates["models_restoration."+nm] = getattr(m, nm)
    except Exception:
        pass
    try:
        m = importlib.import_module("final_model")
        for nm in dir(m):
            if "Restor" in nm or "Unet" in nm:
                candidates["final_model."+nm] = getattr(m, nm)
    except Exception:
        pass
    return candidates

def compute_metrics(gt_pil, restored_pil):
    gt = np.asarray(gt_pil).astype(np.float32)
    out = np.asarray(restored_pil).astype(np.float32)
    if gt.ndim==2: gt = np.stack([gt]*3, -1)
    if out.ndim==2: out = np.stack([out]*3, -1)
    # resize if different
    if gt.shape != out.shape:
        # center-crop or resize gt to out
        from PIL import Image
        out_h, out_w = out.shape[0], out.shape[1]
        out_img = Image.fromarray(out.astype(np.uint8))
        gt_img = Image.fromarray(gt.astype(np.uint8)).resize((out_w, out_h), Image.BILINEAR)
        gt = np.asarray(gt_img)
    psnr = compare_psnr(gt, out, data_range=255.0)
    # compute ssim per-channel average if color
    if gt.ndim==3 and gt.shape[2]==3:
        ssim = 0.0
        for c in range(3):
            ssim += compare_ssim(gt[:,:,c], out[:,:,c], data_range=255.0)
        ssim /= 3.0
    else:
        ssim = compare_ssim(gt, out, data_range=255.0)
    return {"psnr": float(psnr), "ssim": float(ssim)}

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(args.models_dir)
    ckpt_path = Path(args.checkpoint)
    input_folder = Path(args.input_folder)
    out_folder = Path(args.out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    # 1) inspect checkpoint
    print("Loading checkpoint (cpu only) ->", ckpt_path)
    ckpt = safe_load_ckpt(ckpt_path)
    # Detect state_dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            # maybe the checkpoint IS a state_dict (mapping)
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                sd = ckpt
            else:
                sd = None
    else:
        sd = None

    print("Checkpoint keys:", list(ckpt.keys())[:20] if isinstance(ckpt, dict) else type(ckpt))

    # 2) try to instantiate known model classes
    candidates = try_instantiate_known_models()
    print("Model classes discovered in repo:", list(candidates.keys()))
    model = None
    loaded = False
    if sd and candidates:
        # try each candidate by attempting to load state_dict (cpu)
        for name, cls in candidates.items():
            try:
                print("Trying to instantiate", name)
                # try to create a model with default constructor
                m = cls()
                # attempt to load state dict (strip module prefix if needed)
                sdict = sd if isinstance(sd, dict) else sd
                try:
                    m.load_state_dict(sdict)
                except RuntimeError as re:
                    # try stripping module prefix
                    try:
                        m.load_state_dict(strip_module_prefix(sdict))
                    except Exception:
                        raise re
                model = m.to(device)
                loaded = True
                print("Loaded state_dict into", name)
                break
            except Exception as e:
                print("  failed to load into", name, "->", e)
    if not loaded:
        # if checkpoint contains a full model object (rare), try to use it directly
        if isinstance(ckpt, dict) and "model" in ckpt and torch.is_tensor(ckpt["model"])==False:
            print("Checkpoint contains 'model' (non-tensor). Attempting to use direct object.")
            try:
                model = ckpt["model"]
                loaded = True
            except Exception as e:
                print("Could not use 'model' object:", e)

    if not loaded:
        print("No model class could be matched and loaded. Using passthrough fallback (identity).")
        # define a trivial identity model to avoid pipeline crash
        class IdentityModel(torch.nn.Module):
            def forward(self, x):
                return x
        model = IdentityModel().to(device)

    model.eval()

    # 3) collect input images
    imgs = sorted([p for p in input_folder.rglob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]])
    if len(imgs)==0:
        print("No images found in", input_folder)
        return
    n = args.num_samples if args.num_samples>0 else min(20, len(imgs))
    imgs = imgs[:n]
    print(f"Running inference on {len(imgs)} images (device={device})")

    summary_rows = []
    for p in tqdm(imgs):
        try:
            pil = Image.open(p).convert("RGB")
        except Exception as e:
            print("Skipping unreadable:", p, e)
            continue
        inp_tensor = to_tensor(pil).unsqueeze(0).to(device)  # 1xCxHxW
        with torch.no_grad():
            out = model(inp_tensor)  # model may expect different input, but this is a best-effort
        # Postprocess outputs: model may return PIL / numpy / tensor or a dict with 'restored'
        restored = None
        if isinstance(out, dict):
            # try common keys
            for k in ("restored","output","pred","image","recon"):
                if k in out:
                    restored = out[k]
                    break
            if restored is None:
                # maybe returned logits/tensor under 'out'
                if "out" in out:
                    restored = out["out"]
                else:
                    # as last resort, try first value
                    try:
                        restored = list(out.values())[0]
                    except Exception:
                        restored = out
        else:
            restored = out

        # If restored is a tensor shaped like batch, get first element
        if torch.is_tensor(restored):
            if restored.ndim==4 and restored.shape[0]==1:
                restored = restored[0]
            # ensure range: try to rescale if values are >1
            rt = restored.detach().cpu()
            # if in range 0..1 assume ToPILImage expects that
            if rt.max() <= 1.01 and rt.min() >= -0.01:
                # convert
                restored_pil = tensor_to_pil(rt)
            else:
                # assume 0..255
                try:
                    arr = rt.cpu().numpy()
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr,0,255).astype(np.uint8)
                    # reorder to HWC if needed
                    if arr.ndim==3:
                        if arr.shape[0] in (1,3):
                            arr = np.transpose(arr, (1,2,0))
                    restored_pil = Image.fromarray(arr)
                except Exception:
                    restored_pil = tensor_to_pil((rt/255.0).clamp(0,1))
        elif isinstance(restored, Image.Image):
            restored_pil = restored
        elif isinstance(restored, np.ndarray):
            if restored.dtype != np.uint8:
                r = np.clip(restored, 0, 1) * 255.0 if restored.max() <= 1.01 else np.clip(restored, 0, 255)
                arr = r.astype(np.uint8)
            else:
                arr = restored
            if arr.ndim==3 and arr.shape[2] in (3,4):
                restored_pil = Image.fromarray(arr)
            else:
                # single channel
                restored_pil = Image.fromarray(arr)
        else:
            # last fallback: try to convert to string (error)
            print("Unsupported restored output type for", p, type(restored))
            restored_pil = pil  # passthrough

        # save restored image
        rel = p.relative_to(input_folder)
        outp = out_folder / rel
        outp.parent.mkdir(parents=True, exist_ok=True)
        try:
            restored_pil.save(outp.with_suffix(".png"))
        except Exception:
            restored_pil.convert("RGB").save(outp.with_suffix(".png"))

        # compute metrics if GT exists
        metrics = {}
        if args.gt_root:
            gt_p = Path(args.gt_root) / rel
            if gt_p.exists():
                try:
                    gt_pil = Image.open(gt_p).convert("RGB")
                    metrics = compute_metrics(gt_pil, restored_pil)
                except Exception as e:
                    print("Metric compute failed for", p, e)
        summary_rows.append({
            "input": str(p),
            "restored": str(outp.with_suffix(".png")),
            "psnr": metrics.get("psnr",""),
            "ssim": metrics.get("ssim","")
        })

    # save summary CSV
    csvp = out_folder / "validation_summary.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["input","restored","psnr","ssim"])
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    print("Saved restored images + summary to", out_folder, csvp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to .pt checkpoint to validate")
    parser.add_argument("--models-dir", default="./models", help="where .pt models are (not strictly required)")
    parser.add_argument("--input-folder", required=True, help="folder with images to restore")
    parser.add_argument("--out-folder", required=True, help="where to save restored images and summary")
    parser.add_argument("--gt-root", default="", help="optional ground-truth root (same relative paths as input)")
    parser.add_argument("--num-samples", type=int, default=16, help="how many images to test (default 16)")
    args = parser.parse_args()
    main(args)
