import os, json, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.model_segmentation import UNetSeg
from src.utils_segmentation import (
    dice_loss, compute_all_metrics
)

class SegDataset(Dataset):
    def __init__(self, list_file):
        with open(list_file) as f:
            self.pairs = [l.strip().split() for l in f]
        self.tf = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return self.tf(img), self.tf(mask)

def fit_seg(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetSeg().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']))
    bce = torch.nn.BCELoss()

    train_loader = DataLoader(SegDataset(cfg["train_list"]), batch_size=cfg["batch"], shuffle=True)
    val_loader = DataLoader(SegDataset(cfg["val_list"]), batch_size=cfg["batch"], shuffle=False)

    os.makedirs(cfg["model_dir"], exist_ok=True)
    os.makedirs(cfg["result_dir"], exist_ok=True)

    metrics_log = []

    for ep in range(cfg["epochs"]):
        model.train(); loss_sum=0
        for x,y in tqdm(train_loader, desc=f"Epoch {ep+1}/{cfg['epochs']}"):
            x,y=x.to(device),y.to(device)
            opt.zero_grad()
            out=model(x)
            loss=bce(out,y)+dice_loss(out,y)
            loss.backward(); opt.step()
            loss_sum+=loss.item()
        loss_sum/=len(train_loader)

        model.eval(); val_loss=0; agg={}
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device)
                out=model(x)
                val_loss+=bce(out,y).item()
                m=compute_all_metrics(out,y)
                for k,v in m.items(): agg[k]=agg.get(k,0)+v
        val_loss/=len(val_loader)
        for k in agg: agg[k]/=len(val_loader)
        agg.update({"train_loss":loss_sum,"val_loss":val_loss,"epoch":ep+1})
        metrics_log.append(agg)
        print(f"Epoch {ep+1}: IoU={agg['IoU']:.3f} F1={agg['F1']:.3f} Dice={agg['Dice']:.3f}")

    torch.save(model.state_dict(), os.path.join(cfg["model_dir"],"segmentation_unet.pt"))
    with open(os.path.join(cfg["result_dir"],"segmentation_metrics.json"),"w") as f:
        json.dump(metrics_log[-1],f,indent=2)

    plt.plot([m["train_loss"] for m in metrics_log],label="Train")
    plt.plot([m["val_loss"] for m in metrics_log],label="Val")
    plt.legend(); plt.title("Loss"); plt.savefig(os.path.join(cfg["result_dir"],"loss_curve.png")); plt.close()
    plt.plot([m["IoU"] for m in metrics_log],label="IoU")
    plt.plot([m["F1"] for m in metrics_log],label="F1")
    plt.legend(); plt.title("IoU & F1"); plt.savefig(os.path.join(cfg["result_dir"],"iou_f1_curve.png")); plt.close()
    print("✅ Saved model + metrics.")
    return metrics_log[-1]
