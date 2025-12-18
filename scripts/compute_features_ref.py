import os
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms

from src.model_vgg import build_vgg16
from src.utils_ela import generate_ela

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([transforms.ToTensor()])

model = build_vgg16(num_classes=2)
model.load_state_dict(torch.load("models/vgg16_baseline.pt", map_location=device))
model.to(device)
model.eval()

CASIA_TRAIN = "data/processed/ELA/train/authentic"

features = []

def extract_features(x):
    feats = model.model.features(x)
    pooled = torch.nn.functional.adaptive_avg_pool2d(feats, (1,1)).view(x.size(0), -1)
    return pooled.cpu().numpy()

images = os.listdir(CASIA_TRAIN)

for img_name in tqdm(images):
    path = os.path.join(CASIA_TRAIN, img_name)
    ela_np = generate_ela(path)
    t = transform(ela_np).unsqueeze(0).to(device)
    feat = extract_features(t)
    features.append(feat[0])

features = np.array(features)
os.makedirs("data/monitor", exist_ok=True)
np.save("data/monitor/features_ref.npy", features)

print("Saved reference features.")
