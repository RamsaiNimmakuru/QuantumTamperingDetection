# scripts/test_convert.py
from PIL import Image
import numpy as np
import torch
import os
import sys
# import helper from the pipeline file
sys.path.append(os.path.dirname(__file__))
from pipeline_restore_and_classify import to_pil_image_safe

def run_tests():
    pil = Image.new("RGB", (32,32), color=(10, 20, 30))
    print("PIL ->", type(to_pil_image_safe(pil)))

    arr = (np.random.rand(32,32,3)).astype(np.float32)
    print("ndarray float HWC ->", type(to_pil_image_safe(arr)))

    arr_chw = (np.random.rand(3,32,32) * 255).astype(np.uint8)
    print("ndarray CHW uint8 ->", type(to_pil_image_safe(arr_chw)))

    t = torch.rand(3,32,32)
    print("torch CHW float ->", type(to_pil_image_safe(t)))

    tb = torch.rand(1,3,32,32)
    print("torch batched ->", type(to_pil_image_safe(tb)))

if __name__ == "__main__":
    run_tests()
