import os
import random
import cv2
import numpy as np

from basicsr.data.base_dataset import BaseDataset
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import augment

@DATASET_REGISTRY.register()
class PairedSiameseImageDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.io_backend_opt = opt['io_backend']
        self.opt = opt

        self.paths_gt, self.paths_lq_a, self.paths_lq_b = [], [], []
        with open(opt['meta_info_file'], 'r') as f:
            for line in f:
                gt, lq_a, lq_b = line.strip().split(',')
                gt_path = os.path.join(opt['dataroot_gt'], gt.strip())
                lq_a_path = os.path.join(opt['dataroot_lq_a'], lq_a.strip())
                lq_b_path = os.path.join(opt['dataroot_lq_b'], lq_b.strip())

                if not os.path.isfile(gt_path):
                    print(f"[ERROR] GT file not found: {gt_path}")
                    continue
                if not os.path.isfile(lq_a_path):
                    print(f"[ERROR] LQ_A file not found: {lq_a_path}")
                    continue
                if not os.path.isfile(lq_b_path):
                    print(f"[ERROR] LQ_B file not found: {lq_b_path}")
                    continue

                self.paths_gt.append(gt_path)
                self.paths_lq_a.append(lq_a_path)
                self.paths_lq_b.append(lq_b_path)

        if not self.paths_gt:
            raise ValueError("[ERROR] No valid image triplets found in meta_info_file!")

        print(f"[INFO] IO Backend: {self.io_backend_opt}")
        print(f"[INFO] Tổng số ảnh: {len(self.paths_gt)}")
        self.phase = opt.get('phase', 'train')
        self.gt_size = opt.get('gt_size', None)
        self.mean = opt.get('mean', [0.0, 0.0, 0.0])
        self.std = opt.get('std', [1.0, 1.0, 1.0])
        self.use_flip = opt.get('use_flip', True)
        self.use_rot = opt.get('use_rot', True)
        self.file_client = None

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
            print(f"[DEBUG] Initialized FileClient with backend: {self.io_backend_opt}")

        gt_path = self.paths_gt[index]
        lq_a_path = self.paths_lq_a[index]
        lq_b_path = self.paths_lq_b[index]
        print(f"[DEBUG] Loading images: GT={gt_path}, LQ_A={lq_a_path}, LQ_B={lq_b_path}")

        # Load image bytes
        try:
            gt_bytes = self.file_client.get(gt_path, 'gt')
            lq_a_bytes = self.file_client.get(lq_a_path, 'lq_a')
            lq_b_bytes = self.file_client.get(lq_b_path, 'lq_b')
            print(f"[DEBUG] Loaded bytes: GT={len(gt_bytes)} bytes, LQ_A={len(lq_a_bytes)} bytes, LQ_B={len(lq_b_bytes)} bytes")
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to load image files: {str(e)} (GT: {gt_path}, LQ_A: {lq_a_path}, LQ_B: {lq_b_path})")

        # Decode images
        try:
            img_gt = imfrombytes(gt_bytes, float32=True)
            img_lq_a = imfrombytes(lq_a_bytes, float32=True)
            img_lq_b = imfrombytes(lq_b_bytes, float32=True)
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to decode images: {str(e)} (GT: {gt_path}, LQ_A: {lq_a_path}, LQ_B: {lq_b_path})")

        # Check if images are valid
        for name, img, path in [('GT', img_gt, gt_path), ('LQ_A', img_lq_a, lq_a_path), ('LQ_B', img_lq_b, lq_b_path)]:
            if img is None or img.size == 0:
                raise ValueError(f"[ERROR] Invalid or empty image {name}: {path}")
            print(f"[DEBUG] {name} shape: {img.shape}, dtype: {img.dtype} ({path})")

        # Verify image channels (ensure RGB)
        for name, img, path in [('GT', img_gt, gt_path), ('LQ_A', img_lq_a, lq_a_path), ('LQ_B', img_lq_b, lq_b_path)]:
            if len(img.shape) != 3 or img.shape[2] != 3:
                raise ValueError(f"[ERROR] Image {name} is not RGB (shape: {img.shape}): {path}")

        if self.phase == 'train' and self.gt_size is not None:
            h, w = img_gt.shape[:2]
            if h < self.gt_size or w < self.gt_size:
                raise ValueError(f"[ERROR] Image too small to crop: {gt_path} ({h}x{w})")

            rnd_h = random.randint(0, h - self.gt_size)
            rnd_w = random.randint(0, w - self.gt_size)
            img_gt = img_gt[rnd_h:rnd_h + self.gt_size, rnd_w:rnd_w + self.gt_size, :]
            img_lq_a = img_lq_a[rnd_h:rnd_h + self.gt_size, rnd_w:rnd_w + self.gt_size, :]
            img_lq_b = img_lq_b[rnd_h:rnd_h + self.gt_size, rnd_w:rnd_w + self.gt_size, :]
            print(f"[DEBUG] Cropped images to {self.gt_size}x{self.gt_size}: GT shape={img_gt.shape}, LQ_A shape={img_lq_a.shape}, LQ_B shape={img_lq_b.shape}")

            img_gt, img_lq_a, img_lq_b = augment([img_gt, img_lq_a, img_lq_b], self.use_flip, self.use_rot)

        # Convert to tensor
        try:
            img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
            img_lq_a = img2tensor(img_lq_a, bgr2rgb=True, float32=True)
            img_lq_b = img2tensor(img_lq_b, bgr2rgb=True, float32=True)
            print(f"[DEBUG] Converted to tensors: GT shape={img_gt.shape}, LQ_A shape={img_lq_a.shape}, LQ_B shape={img_lq_b.shape}")
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to convert to tensor: {str(e)} (GT: {gt_path}, LQ_A: {lq_a_path}, LQ_B: {lq_b_path})")

        # Normalize
        for t, name in [(img_gt, 'GT'), (img_lq_a, 'LQ_A'), (img_lq_b, 'LQ_B')]:
            for c in range(3):
                t[c, :, :] = (t[c, :, :] - self.mean[c]) / self.std[c]
            print(f"[DEBUG] Normalized {name} tensor: min={t.min().item()}, max={t.max().item()}")

        return {
            'gt': img_gt,
            'lq_a': img_lq_a,
            'lq_b': img_lq_b,
            'gt_path': gt_path,
            'lq_a_path': lq_a_path,
            'lq_b_path': lq_b_path
        }

    def __len__(self):
        return len(self.paths_gt)
