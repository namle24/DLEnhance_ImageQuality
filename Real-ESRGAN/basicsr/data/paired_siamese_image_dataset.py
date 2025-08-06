import os
import random
import cv2
import numpy as np
import gc

from basicsr.data.base_dataset import BaseDataset
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import augment

@DATASET_REGISTRY.register()
class PairedSiameseImageDataset(BaseDataset):
    """Dataset for Siamese Real-ESRGAN using triplet (GT, LQ_A, LQ_B) from meta_info_file."""

    def __init__(self, opt):
        super().__init__(opt)
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', [0.0, 0.0, 0.0])
        self.std = opt.get('std', [1.0, 1.0, 1.0])
        self.gt_size = opt.get('gt_size', None)
        self.scale = opt.get('scale', 4)
        if isinstance(self.gt_size, str):
            if self.gt_size.lower() == 'none':
                self.gt_size = None
            else:
                self.gt_size = int(self.gt_size)
        self.use_flip = opt.get('use_hflip', True)
        self.use_rot = opt.get('use_rot', True)
        self.phase = opt.get('phase', 'train')
        self.max_retries = opt.get('max_retries', 5)  # Limit retries to prevent infinite loops

        self.paths_gt = []
        self.paths_lq_a = []
        self.paths_lq_b = []

        meta_file = opt['meta_info_file']
        with open(meta_file, 'r') as f:
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

        self.file_client = None
        print(f'[INFO] PairedSiameseImageDataset: {len(self.paths_gt)} samples loaded.')

    def __getitem__(self, index, retries=0):
        if retries >= self.max_retries:
            raise ValueError(f"[ERROR] Max retries ({self.max_retries}) reached for index {index}")

        try:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
                print(f"[DEBUG] Initialized FileClient with backend: {self.io_backend_opt}")

            gt_path = self.paths_gt[index]
            lq_a_path = self.paths_lq_a[index]
            lq_b_path = self.paths_lq_b[index]
            print(f"[DEBUG] Loading images: GT={gt_path}, LQ_A={lq_a_path}, LQ_B={lq_b_path}")

            # Load images as bytes
            gt_bytes = self.file_client.get(gt_path, 'gt')
            lq_a_bytes = self.file_client.get(lq_a_path, 'lq_a')
            lq_b_bytes = self.file_client.get(lq_b_path, 'lq_b')
            print(f"[DEBUG] Loaded bytes: GT={len(gt_bytes)} bytes, LQ_A={len(lq_a_bytes)} bytes, LQ_B={len(lq_b_bytes)} bytes")

            # Decode images
            img_gt = imfrombytes(gt_bytes, float32=True)
            img_lq_a = imfrombytes(lq_a_bytes, float32=True)
            img_lq_b = imfrombytes(lq_b_bytes, float32=True)

            # Check if images are valid
            for name, img, path in [('GT', img_gt, gt_path), ('LQ_A', img_lq_a, lq_a_path), ('LQ_B', img_lq_b, lq_b_path)]:
                if img is None or img.size == 0:
                    raise ValueError(f"[ERROR] Invalid or empty image {name}: {path}")
                print(f"[DEBUG] {name} shape: {img.shape}, dtype: {img.dtype} ({path})")

            # Verify image channels (ensure RGB)
            for name, img, path in [('GT', img_gt, gt_path), ('LQ_A', img_lq_a, lq_a_path), ('LQ_B', img_lq_b, lq_b_path)]:
                if len(img.shape) != 3 or img.shape[2] != 3:
                    raise ValueError(f"[ERROR] Image {name} is not RGB (shape: {img.shape}): {path}")

            # Resize LQ images to match GT dimensions
            h_gt, w_gt = img_gt.shape[:2]
            h_lq_a, w_lq_a = img_lq_a.shape[:2]
            h_lq_b, w_lq_b = img_lq_b.shape[:2]
            expected_h_lq, expected_w_lq = h_gt // self.scale, w_gt // self.scale
            if (h_lq_a != expected_h_lq or w_lq_a != expected_w_lq or
                h_lq_b != expected_h_lq or w_lq_b != expected_w_lq):
                print(f"[DEBUG] Resizing LQ images to match GT dimensions: GT({h_gt}, {w_gt})")
                img_lq_a = cv2.resize(img_lq_a, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC)
                img_lq_b = cv2.resize(img_lq_b, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC)
                print(f"[DEBUG] Resized LQ_A to {img_lq_a.shape}, LQ_B to {img_lq_b.shape}")

            # Verify resized dimensions
            if img_lq_a.shape[:2] != (h_gt, w_gt) or img_lq_b.shape[:2] != (h_gt, w_gt):
                raise ValueError(f"[ERROR] Resized LQ images do not match GT dimensions: GT({h_gt}, {w_gt}), LQ_A({img_lq_a.shape[:2]}), LQ_B({img_lq_b.shape[:2]})")

            # Crop if training and gt_size is specified
            if self.phase == 'train' and self.gt_size is not None:
                if h_gt < self.gt_size or w_gt < self.gt_size:
                    raise ValueError(f"[ERROR] Image too small to crop: {gt_path} ({h_gt}x{w_gt})")

                # Ensure crop coordinates are valid
                rnd_h = random.randint(0, h_gt - self.gt_size)
                rnd_w = random.randint(0, w_gt - self.gt_size)
                img_gt = img_gt[rnd_h:rnd_h + self.gt_size, rnd_w:rnd_w + self.gt_size, :]
                img_lq_a = img_lq_a[rnd_h:rnd_h + self.gt_size, rnd_w:rnd_w + self.gt_size, :]
                img_lq_b = img_lq_b[rnd_h:rnd_h + self.gt_size, rnd_w:rnd_w + self.gt_size, :]

                # Verify crop sizes
                for name, img, path in [('GT', img_gt, gt_path), ('LQ_A', img_lq_a, lq_a_path), ('LQ_B', img_lq_b, lq_b_path)]:
                    if img.shape[0] != self.gt_size or img.shape[1] != self.gt_size:
                        raise ValueError(f"[ERROR] Invalid crop size for {name}: {img.shape} (expected {self.gt_size}x{self.gt_size}) at {path}")

                print(f"[DEBUG] Cropped images to {self.gt_size}x{self.gt_size}: GT shape={img_gt.shape}, LQ_A shape={img_lq_a.shape}, LQ_B shape={img_lq_b.shape}")

                img_gt, img_lq_a, img_lq_b = augment([img_gt, img_lq_a, img_lq_b], self.use_flip, self.use_rot)

            # Convert to tensors
            img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
            img_lq_a = img2tensor(img_lq_a, bgr2rgb=True, float32=True)
            img_lq_b = img2tensor(img_lq_b, bgr2rgb=True, float32=True)
            print(f"[DEBUG] Converted to tensors: GT shape={img_gt.shape}, LQ_A shape={img_lq_a.shape}, LQ_B shape={img_lq_b.shape}")

            # Verify tensor shapes
            if img_gt.shape != img_lq_a.shape or img_gt.shape != img_lq_b.shape:
                raise ValueError(f"[ERROR] Tensor shape mismatch: GT({img_gt.shape}), LQ_A({img_lq_a.shape}), LQ_B({img_lq_b.shape})")

            # Normalize tensors
            for t, name in [(img_gt, 'GT'), (img_lq_a, 'LQ_A'), (img_lq_b, 'LQ_B')]:
                for c in range(3):
                    t[c, :, :] = (t[c, :, :] - self.mean[c]) / self.std[c]
                print(f"[DEBUG] Normalized {name} tensor: min={t.min().item()}, max={t.max().item()}")

            # Clean up memory
            del gt_bytes, lq_a_bytes, lq_b_bytes
            gc.collect()

            return {
                'gt': img_gt,
                'lq_a': img_lq_a,
                'lq_b': img_lq_b,
                'gt_path': gt_path,
                'lq_a_path': lq_a_path,
                'lq_b_path': lq_b_path
            }

        except Exception as e:
            print(f"[WARNING] Error loading index {index}: {e}")
            # Clean up memory before retry
            gc.collect()
            return self.__getitem__((index + 1) % self.__len__(), retries + 1)

    def __len__(self):
        return len(self.paths_gt)