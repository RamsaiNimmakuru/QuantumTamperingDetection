#!/usr/bin/env python3
# pipeline_restore_and_classify.py
# Patch: improved restoration model discovery + robust loading + padding + ToPILImage guard

import os
import sys
import argparse
import torch
from pathlib import Path
import torchvision.transforms as T
from PIL import Image
import importlib.util
import importlib
import re
from tqdm import tqdm
import math
import torch.nn.functional as F
import numpy as np

# --- helper functions -----------------------------------------------------
def ensure_repo_on_path():
    repo_root = Path(__file__).resolve().parents[1]  # repo root = parent of scripts/
    s = str(repo_root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return repo_root

def load_checkpoint(path, map_location='cpu'):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    ckpt = torch.load(str(path), map_location=map_location)
    return ckpt

def strip_common_prefixes(state_dict):
    """Try stripping known prefixes to match model keys."""
    new = {}
    for k, v in state_dict.items():
        kk = k
        for p in ("module.", "model.", "net.", "state_dict."):
            if kk.startswith(p):
                kk = kk[len(p):]
        new[kk] = v
    return new

def load_module_from_path(path: Path):
    """Dynamically load a python module from a filepath and return the module object (or None)."""
    try:
        name = "_dynmod_" + re.sub(r'[^0-9a-zA-Z]+', '_', str(path))
        spec = importlib.util.spec_from_file_location(name, str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None

def discover_model_classes(search_dirs=None):
    """
    Search for candidate model classes in repo files.
    Returns dict key->class
    """
    candidates = {}
    repo_root = ensure_repo_on_path()
    if search_dirs is None:
        search_dirs = [repo_root / "scripts", repo_root, repo_root / "models", repo_root / "src"]

    # try some well-known module names (fast)
    known_names = ["model_unet", "models_restoration", "final_model", "model_unet_v2", "models.restoration"]
    for name in known_names:
        try:
            m = importlib.import_module(name)
            for attr in dir(m):
                if any(tok in attr.lower() for tok in ("unet", "restor", "restoration", "decoder", "encoder", "resunet")):
                    cls = getattr(m, attr)
                    if isinstance(cls, type):
                        candidates[f"{name}.{attr}"] = cls
        except Exception:
            pass

    # scan python files
    for d in search_dirs:
        try:
            for p in Path(d).rglob("*.py"):
                if "site-packages" in str(p) or "/dist-packages/" in str(p):
                    continue
                mod = load_module_from_path(p)
                if not mod:
                    continue
                for attr in dir(mod):
                    if any(tok in attr.lower() for tok in ("unet", "restor", "restoration", "decoder", "encoder", "resunet")):
                        cls = getattr(mod, attr)
                        if isinstance(cls, type):
                            key = f"{p.name}.{attr}"
                            candidates[key] = cls
        except Exception:
            continue
    return candidates

def try_instantiate_class(cls, device):
    """Try multiple constructor signatures to get an instance."""
    attempts = [
        (),  # no args
        (3, 3),  # in/out channels
        (64, 3)  # some constructors use filters, channels
    ]
    kwargs_attempts = [
        {},
        {"in_channels": 3, "out_channels": 3},
        {"n_channels": 3, "num_channels": 3, "channels": 3},
        {"input_nc": 3, "output_nc": 3},
    ]
    for args in attempts:
        try:
            inst = cls(*args)
            inst.to(device)
            return inst
        except Exception:
            pass
    for kw in kwargs_attempts:
        try:
            inst = cls(**kw)
            inst.to(device)
            return inst
        except Exception:
            pass
    return None

def robust_load_state_dict(model, ckpt):
    # ckpt can be dict with 'state_dict' or be the state_dict itself
    sdict = ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sdict = ckpt["state_dict"]
    if not isinstance(sdict, dict):
        return False, "checkpoint has no state_dict"
    # ensure tensors on CPU for matching & then moved to model device as needed
    # try direct load
    try:
        model.load_state_dict(sdict)
        return True, "loaded direct"
    except Exception as e:
        # try stripping prefixes
        stripped = strip_common_prefixes(sdict)
        try:
            model.load_state_dict(stripped)
            return True, "loaded after strip prefixes"
        except Exception as e2:
            # try matching by intersection (partial)
            model_dict = model.state_dict()
            filtered = {k: v for k, v in stripped.items() if k in model_dict and model_dict[k].shape == v.shape}
            if filtered:
                model_dict.update(filtered)
                try:
                    model.load_state_dict(model_dict)
                    return True, "partial load with matching keys"
                except Exception as e3:
                    return False, f"partial load failed: {e3}"
            # last attempt: try matching by key suffixes (very permissive)
            suffixed = {}
            for mk in model_dict.keys():
                for ck, v in stripped.items():
                    if ck.endswith(mk):
                        if model_dict[mk].shape == v.shape:
                            suffixed[mk] = v
            if suffixed:
                mdict = model.state_dict()
                mdict.update(suffixed)
                try:
                    model.load_state_dict(mdict)
                    return True, "partial load by suffix match"
                except Exception as e4:
                    return False, f"suffix partial load failed: {e4}"
            return False, f"load failed: {e2}"
# -------------------------------------------------------------------------

# ---------- new helpers for restoration I/O robustness -------------------
def pad_tensor_to_multiple(tensor, multiple=32):
    """Pad tensor (C,H,W) using reflect to nearest multiple for H and W."""
    _, h, w = tensor.shape
    new_h = math.ceil(h / multiple) * multiple
    new_w = math.ceil(w / multiple) * multiple
    pad_bottom = new_h - h
    pad_right = new_w - w
    if pad_bottom == 0 and pad_right == 0:
        return tensor, (0, 0, 0, 0)
    # F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
    padded = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode="reflect")
    return padded, (0, pad_right, 0, pad_bottom)

def unpad_tensor(tensor, pad):
    left, right, top, bottom = pad
    _, h, w = tensor.shape
    h_end = h - bottom if bottom > 0 else h
    w_end = w - right if right > 0 else w
    return tensor[:, 0:h_end, 0:w_end]

def tensor_to_pil_safe(tensor):
    """Convert a CHW tensor in 0..1 to PIL.Image safely."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor_to_pil_safe expects torch.Tensor")
    # clamp & convert to uint8
    t = tensor.detach().cpu()
    # If values are in [-1,1] attempt to rescale, otherwise assume [0,1]
    if t.min() < -0.5 and t.max() <= 1.5:
        t = (t + 1.0) / 2.0
    t = t.clamp(0.0, 1.0)
    # if single channel
    if t.shape[0] == 1:
        arr = (t.squeeze(0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    # else assume 3 channels
    arr = (np.transpose(t.numpy(), (1, 2, 0)) * 255).astype(np.uint8)
    return Image.fromarray(arr)

# -------------------------------------------------------------------------

def run_restoration_inference(input_folder, out_folder, restore_ckpt_path=None, device=None, num_samples=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    Path(out_folder).mkdir(parents=True, exist_ok=True)

    # Discover and instantiate model
    model = None
    ckpt = None
    if restore_ckpt_path:
        ckpt_path = Path(restore_ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(str(ckpt_path))
        # load checkpoint to CPU first, we'll move model to device
        ckpt = load_checkpoint(ckpt_path, map_location="cpu")
        # print a small summary of keys (for debugging)
        if isinstance(ckpt, dict):
            sample_keys = list(ckpt.keys())[:5]
            print(f"[restore] Loaded checkpoint (dict) keys sample: {sample_keys}")
        else:
            print(f"[restore] Loaded checkpoint type: {type(ckpt)}")

        # discover classes
        print("[restore] Discovering model classes in repo...")
        candidates = discover_model_classes()
        print(f"[restore] Found {len(candidates)} candidate classes.")
        for name, cls in candidates.items():
            print(f"[restore] Trying candidate {name} -> {cls}")
            try:
                inst = try_instantiate_class(cls, device)
                if inst is None:
                    print(f"[restore] Could not instantiate {name} (constructor mismatch).")
                    continue
                ok, msg = robust_load_state_dict(inst, ckpt)
                if ok:
                    print(f"[restore] Success: loaded checkpoint into {name} ({msg}). Using this model.")
                    model = inst
                    break
                else:
                    print(f"[restore] Failed to load state_dict into {name}: {msg}")
            except Exception as e:
                print(f"[restore] Exception while testing {name}: {e}")
        if model is None:
            # final attempt: try to load into known class name from repo (common)
            try:
                from train_restoration import ResUNetGenerator  # noqa: F401
                inst = try_instantiate_class(ResUNetGenerator, device)
                if inst:
                    ok, msg = robust_load_state_dict(inst, ckpt)
                    if ok:
                        print(f"[restore] Loaded into train_restoration.ResUNetGenerator ({msg}).")
                        model = inst
            except Exception:
                pass

        if model is None:
            print("[restore] No model could be instantiated + loaded from checkpoint. Falling back to identity (pass-through).")
    else:
        print("[restore] No restore_ckpt_path provided — running pass-through.")

    # If no real model, define a passthrough
    if model is None:
        class PassThrough:
            def __call__(self, pil_img):
                return pil_img
        model = PassThrough()
        is_torch_model = False
    else:
        is_torch_model = isinstance(model, torch.nn.Module)
        # ensure model on device
        model.to(device)
        model.eval()

    # Run inference
    in_paths = sorted([p for p in Path(input_folder).glob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg", ".bmp")])
    if num_samples:
        in_paths = in_paths[:int(num_samples)]
    print(f"[restore] Found {len(in_paths)} images in {input_folder}. Running restoration.")

    # default transforms (adjust if your training used normalization)
    transform_in = T.Compose([T.ToTensor()])
    for p in tqdm(in_paths):
        try:
            img = Image.open(p).convert("RGB")
            orig_w, orig_h = img.size

            if is_torch_model:
                # preprocess
                x = transform_in(img)  # C,H,W in [0,1]
                # pad to multiple (avoid U-Net upsampling mismatches)
                padded_x, pad_info = pad_tensor_to_multiple(x, multiple=32)
                x_b = padded_x.unsqueeze(0).to(device)

                with torch.no_grad():
                    out = model(x_b)

                # some models return (out, aux) or dict
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if isinstance(out, dict) and "out" in out:
                    out = out["out"]

                # if out is tensor with batch
                if isinstance(out, torch.Tensor):
                    out_t = out.squeeze(0).cpu()  # C,H,W
                    # unpad
                    out_t = unpad_tensor(out_t, pad_info)
                    # convert to PIL safely
                    try:
                        out_img = tensor_to_pil_safe(out_t)
                    except Exception as e:
                        # fallback: convert via ToPILImage (still guarded)
                        try:
                            out_img = T.ToPILImage()(out_t.clamp(0.0, 1.0))
                        except Exception:
                            out_img = img
                elif isinstance(out, Image.Image):
                    out_img = out
                    # crop to original if necessary
                    if out_img.size != (orig_w, orig_h):
                        out_img = out_img.crop((0, 0, orig_w, orig_h))
                else:
                    # numpy or other fallback
                    try:
                        if isinstance(out, np.ndarray):
                            # handle HWC or CHW
                            if out.ndim == 3 and out.shape[0] in (1, 3):  # CHW
                                arr = np.transpose(out, (1, 2, 0))
                            else:
                                arr = out
                            arr_u8 = (arr * 255).astype(np.uint8) if arr.max() <= 1.5 else arr.astype(np.uint8)
                            out_img = Image.fromarray(arr_u8)
                            if out_img.size != (orig_w, orig_h):
                                out_img = out_img.crop((0, 0, orig_w, orig_h))
                        else:
                            out_img = img
                    except Exception:
                        out_img = img
            else:
                # passthrough expects PIL
                out_img = model(img)
                # guard types
                if isinstance(out_img, torch.Tensor):
                    # tensor might be CHW in 0..1
                    try:
                        out_img = tensor_to_pil_safe(out_img.squeeze(0) if out_img.ndim == 4 else out_img)
                    except Exception:
                        try:
                            out_img = T.ToPILImage()(out_img.cpu())
                        except Exception:
                            out_img = img
                elif isinstance(out_img, Image.Image):
                    pass
                else:
                    # try numpy
                    if isinstance(out_img, np.ndarray):
                        out_img = Image.fromarray((out_img * 255).astype(np.uint8))
                    else:
                        out_img = img

            # final safety: ensure correct size (crop if necessary)
            if not isinstance(out_img, Image.Image):
                out_img = img
            if out_img.size != (orig_w, orig_h):
                out_img = out_img.crop((0, 0, orig_w, orig_h))

            save_name = Path(out_folder) / p.name
            out_img.save(save_name)
        except Exception as e:
            print(f"[restore] Failed on {p}: {e}")
            # save original as fallback (easy inspection)
            try:
                img = Image.open(p).convert("RGB")
                img.save(Path(out_folder) / ("orig_" + p.name))
            except Exception:
                pass
    return True

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-folder", required=True)
    p.add_argument("--out-restored", required=True)
    p.add_argument("--restore-ckpt", required=False, default=None)
    p.add_argument("--dataset-root", required=False, default=".")
    p.add_argument("--num-samples", required=False, default=None, type=int)
    return p.parse_args()

def main():
    args = parse_args()
    ensure_repo_on_path()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device: {device}")

    # If a checkpoint path was provided, we still try to load it inside run_restoration_inference
    run_restoration_inference(
        input_folder=args.input_folder,
        out_folder=args.out_restored,
        restore_ckpt_path=args.restore_ckpt,
        device=device,
        num_samples=args.num_samples
    )

if __name__ == "__main__":
    main()
