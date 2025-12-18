# gradcam_visualize.py
import os, json, torch, argparse
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from src.model_multistream import MultiStreamFusion

# minimal Grad-CAM: get last conv of rgb branch
def get_cam(model, rgb, ela, res, target_class=1):
    # hook to capture features & grads
    activations = []
    grads = []
    def forward_hook(module, input, output):
        activations.append(output.detach())
    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0].detach())
    # try registering on model.rgb.features[-1] or last conv
    last_conv = None
    for m in reversed(list(model.rgb.features.modules())):
        if hasattr(m, 'weight') and m.weight.ndim==4:
            last_conv = m
            break
    if last_conv is None:
        raise RuntimeError("No conv found in backbone")
    h_fwd = last_conv.register_forward_hook(forward_hook)
    h_bwd = last_conv.register_backward_hook(backward_hook)
    model.zero_grad()
    out = model(rgb, ela, res)
    prob = F.softmax(out, dim=1)[0, target_class]
    prob.backward()
    feat = activations[0][0].cpu().numpy()  # C,H,W
    g = grads[0][0].cpu().numpy()  # C,H,W
    weights = g.mean(axis=(1,2))
    cam = np.zeros(feat.shape[1:], dtype=np.float32)
    for i,w in enumerate(weights):
        cam += w*feat[i]
    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    if cam.max()>0: cam = cam / cam.max()
    h_fwd.remove(); h_bwd.remove()
    return cam

def preprocess_pil(rgb_p, ela_p, base_size=256):
    rgb = Image.open(rgb_p).convert('RGB').resize((base_size,base_size))
    ela = Image.open(ela_p).convert('L')
    ela_rgb = Image.merge('RGB',(ela,ela,ela)).resize((base_size,base_size))
    # center crop 224
    x0=(base_size-224)//2; y0=(base_size-224)//2
    rgb = rgb.crop((x0,y0,x0+224,y0+224))
    ela_rgb = ela_rgb.crop((x0,y0,x0+224,y0+224))
    to_tensor = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))])
    rgb_t = to_tensor(rgb).unsqueeze(0)
    ela_t = to_tensor(ela_rgb).unsqueeze(0)
    # residual
    from scipy.ndimage import gaussian_filter
    rgb_np = np.asarray(rgb).astype(np.float32)
    blurred = gaussian_filter(rgb_np, sigma=(1,1,0))
    residual = np.clip((rgb_np-blurred)/255.0, -1.0, 1.0).transpose(2,0,1)
    res_t = torch.from_numpy(residual).unsqueeze(0).float()
    return rgb_t, ela_t, res_t, rgb

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiStreamFusion(pretrained_rgb=False)
    ck = torch.load(args.ckpt, map_location='cpu')
    if 'model_state' in ck: model.load_state_dict(ck['model_state'])
    else: model.load_state_dict(ck)
    model.to(device).eval()
    os.makedirs(args.out_dir, exist_ok=True)
    # build pairs same as train script
    pairs=[]
    for r,_,fn in os.walk(args.ela_root):
        for f in fn:
            if f.lower().endswith(('.png','.jpg','.jpeg')):
                ela_p = os.path.join(r,f)
                base = os.path.splitext(f)[0]
                # find rgb
                for rr,_,ff in os.walk(args.rgb_root):
                    for rf in ff:
                        if os.path.splitext(rf)[0]==base:
                            rgb_p = os.path.join(rr,rf)
                            pairs.append((rgb_p, ela_p))
                            break
                    else:
                        continue
                    break
    # iterate a few
    cnt=0
    for rgb_p, ela_p in pairs[:args.max_images]:
        rgb_t, ela_t, res_t, rgb_pil = preprocess_pil(rgb_p, ela_p, base_size=args.base_size)
        rgb_t = rgb_t.to(device); ela_t = ela_t.to(device); res_t = res_t.to(device)
        with torch.no_grad():
            out = model(rgb_t, ela_t, res_t)
            prob = F.softmax(out, dim=1)[0,1].item()
            pred = 1 if prob>=0.5 else 0
        cam = get_cam(model, rgb_t, ela_t, res_t, target_class=1)
        # resize cam to rgb_pil size (224x224) and overlay
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        heatmap = cm.jet(cam)[:,:,:3]
        heatmap = (heatmap * 255).astype('uint8')
        heat_pil = Image.fromarray(heatmap).resize(rgb_pil.size)
        overlay = Image.blend(rgb_pil.convert('RGBA'), heat_pil.convert('RGBA'), alpha=0.5)
        fname = f"{cnt}_pred{pred}_p{prob:.3f}.png"
        overlay.save(os.path.join(args.out_dir, fname))
        cnt+=1
    print("Saved overlays to", args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="models/multistream_finetuned_corruptions_best.pt")
    parser.add_argument("--rgb-root", type=str, default="data/raw")
    parser.add_argument("--ela-root", type=str, default="data/processed/ELA")
    parser.add_argument("--out-dir", type=str, default="results/gradcam")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--base-size", type=int, default=256)
    args = parser.parse_args()
    main(args)
