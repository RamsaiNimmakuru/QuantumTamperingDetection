import os, random, torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from src.model_segmentation import UNetSeg

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetSeg().to(device)
model.load_state_dict(torch.load("models/segmentation_unet.pt", map_location=device))
model.eval()

tf = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

with open("data/processed/localization/val.txt") as f:
    samples = random.sample(f.readlines(), 5)

os.makedirs("results/segmentation/visual_samples", exist_ok=True)
for s in samples:
    img_path, mask_path = s.strip().split()
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad(): pred = model(x)[0,0].cpu()
    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1); plt.imshow(img); plt.title("Original"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(mask, cmap='gray'); plt.title("Ground Truth"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(pred, cmap='inferno'); plt.title("Prediction"); plt.axis("off")
    out_path = f"results/segmentation/visual_samples/{os.path.basename(img_path)}"
    plt.savefig(out_path, bbox_inches="tight"); plt.close()
print("✅ Visualization samples saved → results/segmentation/visual_samples/")
