# scripts/train_restoration.py
import os, json, argparse, time, math
from pathlib import Path
from PIL import Image
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.models_restoration import ResUNetGenerator, PatchDiscriminator
from src.perceptual_loss import VGGPerceptualLoss

class PairedDataset(Dataset):
    def __init__(self, root_pairs, size=256):
        clean_dir = Path(root_pairs) / "clean"
        deg_dir = Path(root_pairs) / "degraded"
        # build mapping by stem
        clean_map = {p.stem: p for p in clean_dir.glob("*.*")}
        self.items = []
        for p in deg_dir.glob("*.*"):
            if p.stem in clean_map:
                self.items.append((str(p), str(clean_map[p.stem])))
        self.tf = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        dpath, cpath = self.items[idx]
        d = Image.open(dpath).convert('RGB')
        c = Image.open(cpath).convert('RGB')
        return self.tf(d), self.tf(c)

def compute_psnr_tensor(a, b):
    a = ((a + 1) * 0.5).cpu().numpy()
    b = ((b + 1) * 0.5).cpu().numpy()
    mse = ((a - b) ** 2).mean()
    if mse == 0:
        return float('inf')
    return float(20 * math.log10(1.0 / math.sqrt(mse)))

def save_img_tensor(tensor, out_path):
    img = ((tensor + 1) * 127.5).permute(1,2,0).cpu().numpy().astype('uint8')
    Image.fromarray(img).save(out_path)

def train_epoch(gen, disc, loader, optim_g, optim_d, device, scaler, loss_fns, adv_weight):
    gen.train()
    total = 0.0
    for deg, clean in tqdm(loader, desc="train"):
        deg = deg.to(device); clean = clean.to(device)

        # Generator step
        if scaler:
            with torch.cuda.amp.autocast():
                fake = gen(deg)
                l_l1 = loss_fns['l1'](fake, clean)
                l_perc = loss_fns['perc'](fake, clean)
                if disc:
                    out_fake = disc(torch.cat([deg, fake], dim=1))
                    l_adv = -out_fake.mean()
                else:
                    l_adv = torch.tensor(0.0, device=device)
                lossG = 100.0 * l_l1 + 1.0 * l_perc + adv_weight * l_adv
            optim_g.zero_grad()
            scaler.scale(lossG).backward()
            scaler.step(optim_g)
        else:
            fake = gen(deg)
            l_l1 = loss_fns['l1'](fake, clean)
            l_perc = loss_fns['perc'](fake, clean)
            if disc:
                out_fake = disc(torch.cat([deg, fake], dim=1))
                l_adv = -out_fake.mean()
            else:
                l_adv = torch.tensor(0.0, device=device)
            lossG = 100.0 * l_l1 + 1.0 * l_perc + adv_weight * l_adv
            optim_g.zero_grad(); lossG.backward(); optim_g.step()

        # Discriminator step
        if disc:
            optim_d.zero_grad()
            real_logits = disc(torch.cat([deg, clean], dim=1))
            fake_logits = disc(torch.cat([deg, fake.detach()], dim=1))
            lossD = (nn.ReLU()(1.0 - real_logits)).mean() + (nn.ReLU()(1.0 + fake_logits)).mean()
            if scaler:
                scaler.scale(lossD).backward()
                scaler.step(optim_d)
                scaler.update()
            else:
                lossD.backward(); optim_d.step()

        total += float(lossG.detach().cpu().item())
    return total / max(len(loader), 1)

@torch.no_grad()
def validate(gen, loader, device, out_dir=None, max_samples=50):
    gen.eval()
    psnrs = []
    os.makedirs(out_dir, exist_ok=True) if out_dir else None
    count = 0
    for i, (deg, clean) in enumerate(tqdm(loader, desc="val")):
        deg = deg.to(device); clean = clean.to(device)
        fake = gen(deg)
        for b in range(fake.shape[0]):
            psnrs.append(compute_psnr_tensor(fake[b].cpu(), clean[b].cpu()))
            if out_dir and count < max_samples:
                save_img_tensor(deg[b].cpu(), os.path.join(out_dir, f"{i}_{b}_deg.png"))
                save_img_tensor(fake[b].cpu(), os.path.join(out_dir, f"{i}_{b}_rec.png"))
                save_img_tensor(clean[b].cpu(), os.path.join(out_dir, f"{i}_{b}_tgt.png"))
            count += 1
    avg_psnr = float(np.mean(psnrs)) if len(psnrs) > 0 else 0.0
    return {"psnr": avg_psnr, "samples": int(count)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs", required=True)
    parser.add_argument("--val_pairs", required=True)
    parser.add_argument("--out_dir", default="results/restoration")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--use_gan", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--adv_weight", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = ResUNetGenerator().to(device)
    disc = PatchDiscriminator(in_ch=6).to(device) if args.use_gan else None

    # optionally load pretrain init if exists (file path can be edited)
    pretrain_path = os.path.join("results", "restoration_pretrain", "best_gen.pth")
    if os.path.exists(pretrain_path):
        try:
            gen.load_state_dict(torch.load(pretrain_path, map_location=device), strict=False)
            print("Loaded pretrain init:", pretrain_path)
        except Exception as e:
            print("Warning: couldn't load pretrain init:", e)

    gen_opt = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    disc_opt = optim.Adam(disc.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999)) if disc else None
    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None

    train_ds = PairedDataset(args.train_pairs, size=args.size)
    val_ds = PairedDataset(args.val_pairs, size=args.size)
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dval = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    loss_fns = {'l1': nn.L1Loss().to(device), 'perc': VGGPerceptualLoss(device=device)}
    os.makedirs(args.out_dir, exist_ok=True)
    best_psnr = -1.0

    # save config
    with open(os.path.join(args.out_dir, "train_config.json"), "w") as fh:
        json.dump({"lr": args.lr, "batch_size": args.batch_size, "size": args.size, "use_gan": bool(args.use_gan), "adv_weight": float(args.adv_weight)}, fh, indent=2)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(gen, disc, dl, gen_opt, disc_opt, device, scaler, loss_fns, args.adv_weight)
        val_metrics = validate(gen, dval, device, out_dir=os.path.join(args.out_dir, "samples"), max_samples=50)

        ckpt = {"gen": gen.state_dict(), "disc": disc.state_dict() if disc else None, "epoch": epoch + 1}
        torch.save(ckpt, os.path.join(args.out_dir, f"ckpt_epoch_{epoch+1}.pth"))

        metrics = {"epoch": int(epoch+1), "train_loss": float(train_loss), "val_psnr": float(val_metrics["psnr"]), "val_samples": int(val_metrics["samples"])}
        with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)

        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            torch.save(gen.state_dict(), os.path.join(args.out_dir, "best_gen.pth"))

        print(f"Epoch {epoch+1}/{args.epochs} - train_loss {train_loss:.4f} val_psnr {val_metrics['psnr']:.4f} time {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
