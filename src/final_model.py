import torch
from src.model_unet import UNet
from src.model_segmentation import UNetSeg
from src.model_quantum import HybridQuantumCNN

class FinalTamperDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.restoration = UNet().to(self.device)
        self.localization = UNetSeg().to(self.device)
        self.quantum_model = HybridQuantumCNN().to(self.device)

        self.restoration.load_state_dict(torch.load("models/restoration_unet.pt", map_location=self.device))
        self.localization.load_state_dict(torch.load("models/segmentation_unet.pt", map_location=self.device))
        self.quantum_model.load_state_dict(torch.load("models/quantum_cnn.pt", map_location=self.device))

        self.restoration.eval(); self.localization.eval(); self.quantum_model.eval()

    def analyze(self, img_path):
        from PIL import Image
        from torchvision import transforms
        import matplotlib.pyplot as plt

        tf = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
        img = Image.open(img_path).convert("RGB")
        x = tf(img).unsqueeze(0).to(self.device)

        restored = self.restoration(x)
        mask = self.localization(x)
        result = self.quantum_model(x)
        pred = torch.argmax(result, dim=1).item()

        plt.figure(figsize=(9,3))
        plt.subplot(1,3,1); plt.imshow(img); plt.title("Original"); plt.axis("off")
        plt.subplot(1,3,2); plt.imshow(restored[0].permute(1,2,0).detach().cpu()); plt.title("Restored"); plt.axis("off")
        plt.subplot(1,3,3); plt.imshow(mask[0,0].detach().cpu(), cmap="inferno"); plt.title("Tamper Mask"); plt.axis("off")
        plt.show()

        return {
            "authenticity": "Tampered" if pred == 1 else "Authentic",
            "confidence": torch.softmax(result, dim=1).max().item() * 100
        }
