import os
import random
import cv2
import numpy as np
import torch
from typing import Optional, Tuple, List
from basicsr.data.base_dataset import BaseDataset
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import augment

@DATASET_REGISTRY.register()
class PairedSiameseImageDataset(BaseDataset):
    """Enhanced dataset for Siamese Real-ESRGAN with triplets (GT, LQ_A, LQ_B)."""

    def __init__(self, opt):
        super().__init__(opt)
        # Initialize with error handling
        try:
            self.io_backend_opt = opt['io_backend']
            self.mean = np.array(opt.get('mean', [0.0, 0.0, 0.0]), dtype=np.float32)
            self.std = np.array(opt.get('std', [1.0, 1.0, 1.0]), dtype=np.float32)
            self.gt_size = self._parse_gt_size(opt.get('gt_size'))
            self.scale = opt.get('scale', 4)
            self.use_flip = opt.get('use_hflip', True)
            self.use_rot = opt.get('use_rot', True)
            self.phase = opt.get('phase', 'train')
            self.max_retry = opt.get('max_retry', 3)
            self.skip_corrupted = opt.get('skip_corrupted', True)

            self.paths_gt, self.paths_lq_a, self.paths_lq_b = self._load_meta_info(opt)
            self.file_client = None
            
            print(f'[INFO] Dataset initialized with {len(self.paths_gt)} valid triplets')
        except Exception as e:
            raise RuntimeError(f"Dataset initialization failed: {str(e)}")

    def _parse_gt_size(self, gt_size):
        """Safe GT size parsing"""
        if isinstance(gt_size, str):
            return None if gt_size.lower() == 'none' else int(gt_size)
        return gt_size

    def _load_meta_info(self, opt) -> Tuple[List[str], List[str], List[str]]:
        """Load meta info with validation"""
        paths_gt, paths_lq_a, paths_lq_b = [], [], []
        try:
            with open(opt['meta_info_file'], 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(',')
                    if len(parts) != 3:
                        print(f"[WARN] Invalid line format: {line}")
                        continue

                    gt, lq_a, lq_b = parts
                    gt_path = os.path.abspath(os.path.join(opt['dataroot_gt'], gt.strip()))
                    lq_a_path = os.path.abspath(os.path.join(opt['dataroot_lq_a'], lq_a.strip()))
                    lq_b_path = os.path.abspath(os.path.join(opt['dataroot_lq_b'], lq_b.strip()))

                    if not all(os.path.isfile(p) for p in [gt_path, lq_a_path, lq_b_path]):
                        print(f"[WARN] Missing files in triplet: {gt_path} | {lq_a_path} | {lq_b_path}")
                        continue

                    paths_gt.append(gt_path)
                    paths_lq_a.append(lq_a_path)
                    paths_lq_b.append(lq_b_path)

            if not paths_gt:
                raise ValueError("No valid triplets found in meta_info_file!")
            return paths_gt, paths_lq_a, paths_lq_b
        except Exception as e:
            raise RuntimeError(f"Failed to load meta info: {str(e)}")

    def _load_image(self, path: str, tag: str) -> Optional[np.ndarray]:
        """Safe image loading with retry mechanism"""
        for attempt in range(self.max_retry):
            try:
                img_bytes = self.file_client.get(path, tag)
                img = imfrombytes(img_bytes, float32=True)
                
                if img is None or img.size == 0:
                    raise ValueError(f"Empty image: {path}")
                if len(img.shape) != 3 or img.shape[2] != 3:
                    raise ValueError(f"Image is not RGB: {path}")
                if img.max() > 1.0 or img.min() < 0.0:
                    print(f"[WARN] Image {path} has invalid range [{img.min()}, {img.max()}]")
                return img
            except Exception as e:
                if attempt == self.max_retry - 1:
                    if self.skip_corrupted:
                        print(f"[ERROR] Failed to load {tag} image after {self.max_retry} attempts: {path}")
                        return None
                    raise
                continue
        return None

    def _process_patch(self, img_gt: np.ndarray, img_lq_a: np.ndarray, img_lq_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process and augment image triplet"""
        # Resize LQ images
        h, w = img_gt.shape[:2]
        lq_size = (w // self.scale, h // self.scale)
        img_lq_a = cv2.resize(img_lq_a, lq_size, interpolation=cv2.INTER_AREA)
        img_lq_b = cv2.resize(img_lq_b, lq_size, interpolation=cv2.INTER_AREA)

        # Training-specific processing
        if self.phase == 'train' and self.gt_size:
            lq_crop_size = self.gt_size // self.scale
            
            # Pad if needed (reflection padding)
            pad_h = max(0, lq_crop_size - img_lq_a.shape[0])
            pad_w = max(0, lq_crop_size - img_lq_a.shape[1])
            if pad_h > 0 or pad_w > 0:
                img_lq_a = cv2.copyMakeBorder(img_lq_a, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                img_lq_b = cv2.copyMakeBorder(img_lq_b, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h * self.scale, 0, pad_w * self.scale, cv2.BORDER_REFLECT)

            # Random crop
            y = random.randint(0, img_lq_a.shape[0] - lq_crop_size)
            x = random.randint(0, img_lq_a.shape[1] - lq_crop_size)
            
            img_lq_a = img_lq_a[y:y+lq_crop_size, x:x+lq_crop_size]
            img_lq_b = img_lq_b[y:y+lq_crop_size, x:x+lq_crop_size]
            img_gt = img_gt[y*self.scale:(y+lq_crop_size)*self.scale, 
                          x*self.scale:(x+lq_crop_size)*self.scale]

        # Augmentation
        img_gt, img_lq_a, img_lq_b = augment([img_gt, img_lq_a, img_lq_b], self.use_flip, self.use_rot)
        return img_gt, img_lq_a, img_lq_b

    def _normalize(self, img: np.ndarray) -> torch.Tensor:
        """Normalize image to tensor"""
        img = img2tensor(img, bgr2rgb=True, float32=True)
        img = (img - torch.from_numpy(self.mean).view(3, 1, 1)) / torch.from_numpy(self.std).view(3, 1, 1)
        return img

    def __getitem__(self, index):
        # Initialize file client if needed
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path = self.paths_gt[index]
        lq_a_path = self.paths_lq_a[index]
        lq_b_path = self.paths_lq_b[index]

        try:
            # Load with error handling
            img_gt = self._load_image(gt_path, 'gt')
            img_lq_a = self._load_image(lq_a_path, 'lq_a')
            img_lq_b = self._load_image(lq_b_path, 'lq_b')
            
            if None in [img_gt, img_lq_a, img_lq_b]:
                if self.skip_corrupted:
                    return self.__getitem__((index + 1) % len(self))
                raise ValueError("Failed to load one or more images")

            # Process images
            img_gt, img_lq_a, img_lq_b = self._process_patch(img_gt, img_lq_a, img_lq_b)

            return {
                'gt': self._normalize(img_gt),
                'lq_a': self._normalize(img_lq_a),
                'lq_b': self._normalize(img_lq_b),
                'gt_path': gt_path,
                'lq_a_path': lq_a_path,
                'lq_b_path': lq_b_path
            }
        except Exception as e:
            print(f"Error processing index {index}: {str(e)}")
            if self.skip_corrupted:
                return self.__getitem__((index + 1) % len(self))
            raise

    def __len__(self):
        return len(self.paths_gt)