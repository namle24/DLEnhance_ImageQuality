import os
import random
import cv2
import numpy as np
from typing import Optional
import torch
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
        self.mean = np.array(opt.get('mean', [0.0, 0.0, 0.0]), dtype=np.float32)
        self.std = np.array(opt.get('std', [1.0, 1.0, 1.0]), dtype=np.float32)
        self.gt_size = opt.get('gt_size', None)
        self.scale = opt.get('scale', 4)
        self.use_flip = opt.get('use_hflip', True)
        self.use_rot = opt.get('use_rot', True)
        self.phase = opt.get('phase', 'train')
        self.max_retry = opt.get('max_retry', 3)  # Max retry attempts for loading
        self.skip_corrupted = opt.get('skip_corrupted', True)  # Skip corrupted images

        # Validate and convert gt_size
        if isinstance(self.gt_size, str):
            self.gt_size = None if self.gt_size.lower() == 'none' else int(self.gt_size)

        self.paths_gt = []
        self.paths_lq_a = []
        self.paths_lq_b = []

        # Load meta file with error handling
        meta_file = opt['meta_info_file']
        try:
            with open(meta_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    parts = line.split(',')
                    if len(parts) != 3:
                        print(f"[WARNING] Invalid line format in meta file: {line}")
                        continue

                    gt, lq_a, lq_b = parts
                    gt_path = os.path.abspath(os.path.join(opt['dataroot_gt'], gt.strip()))
                    lq_a_path = os.path.abspath(os.path.join(opt['dataroot_lq_a'], lq_a.strip()))
                    lq_b_path = os.path.abspath(os.path.join(opt['dataroot_lq_b'], lq_b.strip()))

                    # Check file existence
                    if not all(os.path.isfile(p) for p in [gt_path, lq_a_path, lq_b_path]):
                        print(f"[WARNING] Missing files in triplet: {gt_path}, {lq_a_path}, {lq_b_path}")
                        continue

                    self.paths_gt.append(gt_path)
                    self.paths_lq_a.append(lq_a_path)
                    self.paths_lq_b.append(lq_b_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load meta file {meta_file}: {str(e)}")

        if not self.paths_gt:
            raise ValueError("[ERROR] No valid image triplets found in meta_info_file!")

        self.file_client = None
        print(f'[INFO] PairedSiameseImageDataset: {len(self.paths_gt)} samples loaded.')

    def _load_image(self, path: str, tag: str) -> Optional[np.ndarray]:
        """Safe image loading with retry mechanism."""
        for attempt in range(self.max_retry):
            try:
                img_bytes = self.file_client.get(path, tag)
                img = imfrombytes(img_bytes, float32=True)
                
                if img is None or img.size == 0:
                    raise ValueError(f"Empty image: {path}")
                if len(img.shape) != 3 or img.shape[2] != 3:
                    raise ValueError(f"Image is not RGB: {path}")
                
                return img
            except Exception as e:
                if attempt == self.max_retry - 1:
                    if self.skip_corrupted:
                        print(f"[ERROR] Failed to load {tag} image after {self.max_retry} attempts: {path}")
                        return None
                    raise
                continue
        return None

    def _process_images(self, img_gt: np.ndarray, img_lq_a: np.ndarray, img_lq_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process and augment image triplet."""
        # Calculate expected LQ size
        h_gt, w_gt = img_gt.shape[:2]
        expected_h_lq = h_gt // self.scale
        expected_w_lq = w_gt // self.scale

        # Resize LQ images with proper interpolation
        img_lq_a = cv2.resize(img_lq_a, (expected_w_lq, expected_h_lq), interpolation=cv2.INTER_AREA)
        img_lq_b = cv2.resize(img_lq_b, (expected_w_lq, expected_h_lq), interpolation=cv2.INTER_AREA)

        # Training-specific processing
        if self.phase == 'train' and self.gt_size is not None:
            lq_crop_size = self.gt_size // self.scale
            
            # Pad images if needed
            pad_h = max(0, lq_crop_size - img_lq_a.shape[0])
            pad_w = max(0, lq_crop_size - img_lq_a.shape[1])
            
            if pad_h > 0 or pad_w > 0:
                img_lq_a = cv2.copyMakeBorder(img_lq_a, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                img_lq_b = cv2.copyMakeBorder(img_lq_b, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h * self.scale, 0, pad_w * self.scale, cv2.BORDER_REFLECT)

            # Random crop
            top = random.randint(0, max(0, img_lq_a.shape[0] - lq_crop_size))
            left = random.randint(0, max(0, img_lq_a.shape[1] - lq_crop_size))
            
            img_lq_a = img_lq_a[top:top + lq_crop_size, left:left + lq_crop_size, :]
            img_lq_b = img_lq_b[top:top + lq_crop_size, left:left + lq_crop_size, :]
            img_gt = img_gt[top*self.scale:(top + lq_crop_size)*self.scale, 
                            left*self.scale:(left + lq_crop_size)*self.scale, :]

        # Apply augmentations
        img_gt, img_lq_a, img_lq_b = augment([img_gt, img_lq_a, img_lq_b], self.use_flip, self.use_rot)

        return img_gt, img_lq_a, img_lq_b

    def _normalize_image(self, img: np.ndarray) -> torch.Tensor:
        """Convert and normalize image to tensor."""
        img = img2tensor(img, bgr2rgb=True, float32=True)
        img = (img - torch.tensor(self.mean, device=img.device).view(3, 1, 1)) / \
              torch.tensor(self.std, device=img.device).view(3, 1, 1)
        return img

    def __getitem__(self, index):
        # Initialize file client if needed
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path = self.paths_gt[index]
        lq_a_path = self.paths_lq_a[index]
        lq_b_path = self.paths_lq_b[index]

        try:
            # Load images with error handling
            img_gt = self._load_image(gt_path, 'gt')
            img_lq_a = self._load_image(lq_a_path, 'lq_a')
            img_lq_b = self._load_image(lq_b_path, 'lq_b')

            if None in [img_gt, img_lq_a, img_lq_b]:
                if self.skip_corrupted:
                    # Skip corrupted images by returning next valid item
                    return self.__getitem__((index + 1) % len(self))
                raise ValueError(f"Failed to load one or more images in triplet: {gt_path}")

            # Process images
            img_gt, img_lq_a, img_lq_b = self._process_images(img_gt, img_lq_a, img_lq_b)

            # Convert to tensors
            return {
                'gt': self._normalize_image(img_gt),
                'lq_a': self._normalize_image(img_lq_a),
                'lq_b': self._normalize_image(img_lq_b),
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