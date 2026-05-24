# evaluate_niqe.py
# đặt ở thư mục gốc Real-ESRGAN

import os
import cv2
import glob
import numpy as np
from basicsr.metrics.niqe import calculate_niqe

# thư mục visualization
root_dir = '/data/home/namlh/DLEnhance_ImageQuality/Real-ESRGAN/experiments/train_PD_MAE_RealESRGAN_x4plus_20262405/visualization'
# lấy tất cả subfolder
subfolders = sorted([
    f for f in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, f))
])

if not subfolders:
    print(f'Không tìm thấy folder trong {root_dir}')
    exit()

all_scores = []

for folder in subfolders:
    folder_path = os.path.join(root_dir, folder)

    imgs = sorted(glob.glob(os.path.join(folder_path, '*.png')))

    if not imgs:
        print(f'Không có ảnh trong {folder_path}')
        continue

    print(f'\n===== {folder} =====')

    folder_scores = []

    for path in imgs:
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if img is None:
            print(f'Không đọc được ảnh: {path}')
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        score = calculate_niqe(img, crop_border=4)

        folder_scores.append(score)
        all_scores.append(score)

        print(f'{os.path.basename(path)}: {score:.4f}')

    print(f'\n[{folder}] Mean NIQE: {np.mean(folder_scores):.4f}')
    print(f'[{folder}] Std:       {np.std(folder_scores):.4f}')

print('\n==============================')
print(f'Overall Mean NIQE: {np.mean(all_scores):.4f}')
print(f'Overall Std:       {np.std(all_scores):.4f}')