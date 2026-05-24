# evaluate_niqe_realsr.py

import os
import cv2
import glob
import numpy as np
from basicsr.metrics.niqe import calculate_niqe

# đúng path
dataset_root = '/data/home/namlh/data/benchmark_datasets/RealSR(V3)'

datasets = ['Canon', 'Nikon']

overall_scores = []

for dataset in datasets:

    lr_root = os.path.join(dataset_root, dataset, 'LR')

    if not os.path.isdir(lr_root):
        print(f'Không tìm thấy folder: {lr_root}')
        continue

    print(f'\n==============================')
    print(f'DATASET: {dataset}')
    print('==============================')

    dataset_scores = []

    imgs = sorted(
        glob.glob(os.path.join(lr_root, '**', '*.png'), recursive=True)
    )

    if len(imgs) == 0:
        print(f'Không có ảnh trong {lr_root}')
        continue

    model_scores = {}

    for path in imgs:

        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        score = calculate_niqe(img, crop_border=4)

        dataset_scores.append(score)
        overall_scores.append(score)

        # tên folder cha
        model_name = os.path.basename(os.path.dirname(path))

        if model_name not in model_scores:
            model_scores[model_name] = []

        model_scores[model_name].append(score)

    # mean từng model/state
    for model_name in sorted(model_scores.keys()):
        print(
            f'{model_name:<20} '
            f'Mean NIQE: {np.mean(model_scores[model_name]):.4f}'
        )

    print(f'\n[{dataset}] Mean NIQE: {np.mean(dataset_scores):.4f}')

print('\n==============================')

if len(overall_scores) > 0:
    print(f'TOTAL Mean NIQE: {np.mean(overall_scores):.4f}')
else:
    print('Không có ảnh nào được xử lý.')

print('==============================')