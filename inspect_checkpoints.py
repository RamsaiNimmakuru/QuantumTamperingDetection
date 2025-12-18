# inspect_checkpoints.py
import torch
from pathlib import Path
import argparse
import json

def summarize_tensor(t):
    return {"dtype": str(t.dtype), "shape": list(t.shape), "numel": t.numel()}

def inspect_ckpt(path: Path):
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        return {"file": str(path), "error": f"load_failed: {e}"}

    out = {"file": str(path)}
    if isinstance(ckpt, dict):
        out["keys"] = list(ckpt.keys())
        # try to summarize any large tensors in common fields
        for k in ("state_dict", "model", "net", "params"):
            if k in ckpt:
                sd = ckpt[k]
                if isinstance(sd, dict):
                    out["state_dict_keys_sample"] = list(sd.keys())[:10]
                    # summarize shapes for first 6 tensors
                    small = {}
                    for i,(kk,v) in enumerate(sd.items()):
                        if i>=6: break
                        try:
                            if hasattr(v, "shape"):
                                small[kk] = summarize_tensor(v)
                        except Exception:
                            small[kk] = "not_tensor"
                    out["state_dict_sample_summary"] = small
                break
        # if contains extra info like 'epoch' 'arch' 'config'
        extras = {}
        for name in ("epoch","arch","config","optimizer","args"):
            if name in ckpt:
                extras[name] = type(ckpt[name]).__name__
        if extras:
            out["extras"] = extras
    else:
        out["type"] = type(ckpt).__name__
    return out

def main(folder):
    folder = Path(folder)
    results = []
    for p in sorted(folder.glob("*.pt")):
        print("Inspecting", p.name)
        r = inspect_ckpt(p)
        results.append(r)
    # print summary
    print("\n--- Summary ---")
    for r in results:
        print(r["file"])
        if "error" in r:
            print("  ERROR:", r["error"])
            continue
        print("  keys:", r.get("keys", "N/A"))
        if "state_dict_sample_summary" in r:
            print("  sample state_dict keys/shapes:")
            for k,v in r["state_dict_sample_summary"].items():
                print("    ", k, v)
    # save JSON for later
    outp = folder / "checkpoints_inspect.json"
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved inspection to", outp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default=".", help="folder where .pt files are (default: current dir)")
    args = parser.parse_args()
    main(args.models_dir)
