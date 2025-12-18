import os
import time
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms

from src.model_vgg import build_vgg16
from src.utils_ela import generate_ela

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# CONFIGURATION
# =============================
CONF_THRESH = 0.70
LOWCONF_FRAC = 0.20
FEATURE_SHIFT_THRESH = 0.15

ADAPT_EPOCHS = 3
ADAPT_LR = 5e-5
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4
MIN_SAMPLES_TO_ADAPT = 50
REPLAY_FRACTION = 0.25

MODEL_PATH = "models/vgg16_baseline.pt"
BACKUP_PATH = "models/vgg16_backup_prev.pt"
ADAPT_SAVE_DIR = "models"

transform = transforms.Compose([transforms.ToTensor()])


# ======================================
# LOAD MODEL
# ======================================
def load_model():
    model = build_vgg16(num_classes=2)
    sd = torch.load(MODEL_PATH, map_location="cpu")
    try:
        model.load_state_dict(sd)
    except:
        model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


# ======================================
# LOAD REFERENCE FEATURES
# ======================================
def load_reference_features():
    ref_path = "data/monitor/features_ref.npy"
    if not os.path.exists(ref_path):
        print("❌ Reference features not found. Run compute_features_ref.py first.")
        return None
    feats = np.load(ref_path)
    return feats.mean(axis=0)


# ======================================
# EXTRACT FEATURES FROM MODEL BACKBONE
# ======================================
def extract_features(model, batch_tensor):
    with torch.no_grad():
        x = batch_tensor.to(DEVICE)
        feats = model.model.features(x)
        pooled = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1)).view(x.size(0), -1)
        return pooled.cpu().numpy()


# ======================================
# MONITOR NEW BATCH
# ======================================
def monitor_batch(model, image_paths, ref_centroid):
    probs = []
    features = []
    ela_cache = []

    for p in tqdm(image_paths, desc="Monitoring batch"):
        try:
            ela = generate_ela(p)
            ela_cache.append((p, ela))

            t = transform(ela).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(t)
                prob = torch.softmax(logits, dim=1)[0][1].item()

            probs.append(prob)

            feat = extract_features(model, t)[0]
            features.append(feat)

        except:
            continue

    probs = np.array(probs)
    features = np.array(features)

    # confidence metrics
    mean_conf = probs.mean() if len(probs) > 0 else 0
    lowconf_ratio = (probs < CONF_THRESH).mean() if len(probs) > 0 else 1

    # feature shift
    shift = 0
    if ref_centroid is not None and len(features) > 0:
        curr_centroid = features.mean(axis=0)
        dot = np.dot(ref_centroid, curr_centroid)
        shift = 1 - (dot / (np.linalg.norm(ref_centroid) * np.linalg.norm(curr_centroid) + 1e-9))

    triggered = (mean_conf < CONF_THRESH) or (lowconf_ratio > LOWCONF_FRAC) or (shift > FEATURE_SHIFT_THRESH)

    return {
        "triggered": triggered,
        "mean_conf": mean_conf,
        "lowconf_ratio": lowconf_ratio,
        "shift": shift,
        "probs": probs,
        "features": features,
        "cache": ela_cache
    }


# ======================================
# FINE TUNING
# ======================================
def fine_tune(model, train_samples):

    # unfreeze classifier
    for p in model.model.classifier.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=ADAPT_LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    print("\n🔥 Fine-tuning on new data...")

    for epoch in range(ADAPT_EPOCHS):
        random.shuffle(train_samples)
        batches = [train_samples[i:i + BATCH_SIZE] for i in range(0, len(train_samples), BATCH_SIZE)]

        for batch in batches:
            xs = torch.stack([transform(x[1]) for x in batch]).to(DEVICE)
            ys = torch.tensor([x[2] for x in batch], dtype=torch.long).to(DEVICE)

            optimizer.zero_grad()
            logits = model(xs)
            loss = loss_fn(logits, ys)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{ADAPT_EPOCHS} complete.")

    model.eval()
    return model


# ======================================
# VALIDATION AFTER ADAPTATION
# ======================================
def validate(model, val_samples):
    from sklearn.metrics import f1_score

    true = []
    pred = []

    for (p, ela, label) in val_samples:
        x = transform(ela).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.softmax(model(x), dim=1)[0][1].item()
        true.append(label)
        pred.append(1 if prob > 0.5 else 0)

    return f1_score(true, pred)


# ======================================
# MAIN ADAPTION FUNCTION
# ======================================
def adapt_process(new_images_folder):

    model = load_model()
    ref_centroid = load_reference_features()

    new_imgs = [os.path.join(new_images_folder, f)
                for f in os.listdir(new_images_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))]

    if len(new_imgs) < MIN_SAMPLES_TO_ADAPT:
        print(f"❌ Not enough images in batch ({len(new_imgs)}). Need at least {MIN_SAMPLES_TO_ADAPT}.")
        return

    monitor = monitor_batch(model, new_imgs, ref_centroid)

    print("\nMONITOR REPORT:")
    print("Average confidence :", monitor["mean_conf"])
    print("Low-confidence rate:", monitor["lowconf_ratio"])
    print("Feature shift      :", monitor["shift"])
    print("Triggered          :", monitor["triggered"])

    if not monitor["triggered"]:
        print("\n✅ No adaptation needed. Model remains unchanged.")
        return

    print("\n⚠️ Adaptation Triggered!")

    # Pseudo-label rules
    collected = []
    for (p, ela) in monitor["cache"]:
        t = transform(ela).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.softmax(model(t), dim=1)[0][1].item()

        if prob > 0.95:
            label = 1
        elif prob < 0.05:
            label = 0
        else:
            continue

        collected.append((p, ela, label))

    if len(collected) < MIN_SAMPLES_TO_ADAPT:
        print("❌ Not enough high-confidence pseudo labels. Need human labels.")
        return

    # Split into train/val
    random.shuffle(collected)
    val_count = max(10, len(collected) // 5)
    val_samples = collected[:val_count]
    train_samples = collected[val_count:]

    # Backup old model
    shutil.copy(MODEL_PATH, BACKUP_PATH)

    # Train new model
    model = fine_tune(model, train_samples)

    # Validate
    new_f1 = validate(model, val_samples)
    print("New F1 after adaptation:", new_f1)

    # Load old model to compare
    old_model = load_model()
    old_f1 = validate(old_model, val_samples)
    print("Old F1 before adaptation:", old_f1)

    # Save only if improved
    if new_f1 > old_f1 + 0.01:
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(ADAPT_SAVE_DIR, f"vgg16_adapted_{ts}.pt")
        torch.save(model.state_dict(), save_path)
        print("\n🎉 Adapted model saved as:", save_path)
    else:
        shutil.copy(BACKUP_PATH, MODEL_PATH)
        print("\n❌ Adaptation did not improve performance. Rolled back.")


# ======================================
# ENTRY POINT
# ======================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, required=True,
                        help="Folder containing new images for adaptation")
    args = parser.parse_args()

    adapt_process(args.batch)
