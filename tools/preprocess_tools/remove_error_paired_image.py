import os
import cv2

folders = {
    "HR_sub": "/datasets/RealESRGAN_data/dataset/train/HR_sub",
    "LR_light_sub": "/datasets/RealESRGAN_data/dataset/train/LR_light_sub",
    "LR_moderate_sub": "/datasets/RealESRGAN_data/dataset/train/LR_moderate_sub"
}

crop_size = 120

to_remove = []

hr_folder = folders["HR_sub"]
for fname in os.listdir(hr_folder):
    fpath = os.path.join(hr_folder, fname)
    img = cv2.imread(fpath)
    if img is None:
        print(f"[INVALID] Không đọc được ảnh: {fname}")
        to_remove.append(fname)
        continue
    h, w = img.shape[:2]
    if h < crop_size or w < crop_size:
        print(f"[TOO SMALL] {fname} - size: {w}x{h}")
        to_remove.append(fname)

for folder_name, folder_path in folders.items():
    for fname in to_remove:
        fpath = os.path.join(folder_path, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"[REMOVED] {folder_name}/{fname}")

print(f"\nĐã xoá {len(to_remove)} ảnh nhỏ hơn {crop_size}x{crop_size}")
