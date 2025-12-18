# scripts/infer_restoration.py
import os, argparse
from PIL import Image
import torch
from torchvision import transforms
from src.models_restoration import ResUNetGenerator

def run_infer(model_path, src_dir, out_dir, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = ResUNetGenerator().to(device)
    gen.load_state_dict(torch.load(model_path, map_location="cpu"))
    gen.eval()
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])
    os.makedirs(out_dir, exist_ok=True)
    for fn in sorted(os.listdir(src_dir)):
        if not fn.lower().endswith(('.jpg','.jpeg','.png','.tif')):
            continue
        img = Image.open(os.path.join(src_dir, fn)).convert('RGB')
        inp = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            rec = gen(inp)[0]
        rec_img = ((rec.cpu() + 1) * 127.5).permute(1,2,0).numpy().astype('uint8')
        Image.fromarray(rec_img).save(os.path.join(out_dir, fn))
    print("Saved restored images to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    run_infer(args.model, args.src, args.out, args.size)
