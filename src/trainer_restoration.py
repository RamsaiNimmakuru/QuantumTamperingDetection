import os, json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.dataset_restoration import RestorationDataset
from src.model_unet import UNet
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import numpy as np

def fit_restoration(cfg):
    device="cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"],exist_ok=True)
    os.makedirs(cfg["result_dir"],exist_ok=True)
    os.makedirs(os.path.join(cfg["result_dir"],"restored_samples"),exist_ok=True)

    ds=RestorationDataset(cfg["degraded_dir"],cfg["clean_dir"])
    dl=DataLoader(ds,batch_size=cfg["batch_size"],shuffle=True)
    model=UNet().to(device)
    loss_fn=nn.L1Loss()
    opt=optim.Adam(model.parameters(),lr=cfg["lr"])
    hist=[]

    for ep in range(cfg["epochs"]):
        model.train(); losses=[]
        for x,y in tqdm(dl,desc=f"Epoch {ep+1}/{cfg['epochs']}"):
            x,y=x.to(device),y.to(device)
            opt.zero_grad(); out=model(x)
            loss=loss_fn(out,y); loss.backward(); opt.step()
            losses.append(loss.item())
        ep_loss=np.mean(losses)
        hist.append(ep_loss); print(f"Epoch {ep+1}: loss {ep_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(cfg["save_dir"],"restoration_unet.pt"))

    # Loss plot
    plt.plot(hist); plt.title("Training Loss")
    plt.savefig(os.path.join(cfg["result_dir"],"loss_curve.png")); plt.close()

    # Evaluate sample PSNR/SSIM
    model.eval(); total_psnr, total_ssim = 0, 0
    with torch.no_grad():
        for i,(x,y) in enumerate(dl):
            x,y=x.to(device),y.to(device)
            out=model(x)
            for j in range(len(out)):
                o=out[j].cpu().permute(1,2,0).numpy()
                g=y[j].cpu().permute(1,2,0).numpy()
                total_psnr += psnr(g,o,data_range=1)
                total_ssim += ssim(g,o,channel_axis=2,data_range=1)
                if i==0 and j<3:
                    plt.subplot(3,3,3*j+1); plt.imshow(x[j].cpu().permute(1,2,0)); plt.axis('off'); plt.title('Input')
                    plt.subplot(3,3,3*j+2); plt.imshow(o); plt.axis('off'); plt.title('Restored')
                    plt.subplot(3,3,3*j+3); plt.imshow(g); plt.axis('off'); plt.title('GT')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg["result_dir"],"sample_comparison.png")); plt.close()

    avg_psnr=total_psnr/len(dl.dataset)
    avg_ssim=total_ssim/len(dl.dataset)
    metrics={"psnr":avg_psnr,"ssim":avg_ssim}
    json.dump(metrics, open(os.path.join(cfg["result_dir"],"restoration_metrics.json"),"w"), indent=2)
    print("✅ Restoration training done.")
    return metrics
