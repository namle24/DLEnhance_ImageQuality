import os
import random
import cv2
import numpy as np
import torch
from basicsr.data.base_dataset import BaseDataset
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import augment
from torchvision.transforms.functional import normalize

@DATASET_REGISTRY.register()
class PairedSiameseImageDataset(BaseDataset):
    """Enhanced Dataset for Siamese Real-ESRGAN with better preprocessing."""

    def __init__(self, opt):
        super().__init__(opt)
        self.io_backend_opt = opt['io_backend']
        
        # Proper normalization for Real-ESRGAN (keep images in [0,1] range)
        self.mean = opt.get('mean', None)  # Don't normalize by default
        self.std = opt.get('std', None)
        
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
        
        # Enhanced augmentation options
        self.color_jitter = opt.get('color_jitter', False)
        self.blur_prob = opt.get('blur_prob', 0.1)
        self.noise_prob = opt.get('noise_prob', 0.1)

        self.paths_gt = []
        self.paths_lq_a = []
        self.paths_lq_b = []

        # Load metadata
        meta_file = opt['meta_info_file']
        with open(meta_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 3:
                    continue
                    
                gt, lq_a, lq_b = [p.strip() for p in parts]
                gt_path = os.path.join(opt['dataroot_gt'], gt)
                lq_a_path = os.path.join(opt['dataroot_lq_a'], lq_a)
                lq_b_path = os.path.join(opt['dataroot_lq_b'], lq_b)

                # Verify files exist
                if all(os.path.isfile(p) for p in [gt_path, lq_a_path, lq_b_path]):
                    self.paths_gt.append(gt_path)
                    self.paths_lq_a.append(lq_a_path)
                    self.paths_lq_b.append(lq_b_path)
                else:
                    print(f"[WARNING] Missing files for: {gt}, {lq_a}, {lq_b}")

        if not self.paths_gt:
            raise ValueError("[ERROR] No valid image triplets found in meta_info_file!")

        self.file_client = None
        print(f'[INFO] PairedSiameseImageDataset: {len(self.paths_gt)} samples loaded.')

    def apply_additional_augmentation(self, img_list):
        """Apply additional augmentation during training."""
        if self.phase != 'train':
            return img_list
            
        # Color jitter
        if self.color_jitter and random.random() < 0.3:
            # Random brightness/contrast
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.uniform(-0.1, 0.1)   # brightness
            for i in range(len(img_list)):
                img_list[i] = np.clip(img_list[i] * alpha + beta, 0, 1)
        
        # Random Gaussian blur
        if random.random() < self.blur_prob:
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.1, 1.0)
            for i in range(len(img_list)):
                img_list[i] = cv2.GaussianBlur(img_list[i], (kernel_size, kernel_size), sigma)
        
        # Random noise
        if random.random() < self.noise_prob:
            noise_std = random.uniform(0.01, 0.05)
            for i in range(len(img_list)):
                noise = np.random.normal(0, noise_std, img_list[i].shape)
                img_list[i] = np.clip(img_list[i] + noise, 0, 1)
                
        return img_list

    def __getitem__(self, index):
        try:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

            gt_path = self.paths_gt[index]
            lq_a_path = self.paths_lq_a[index]
            lq_b_path = self.paths_lq_b[index]

            # Load images
            img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)
            img_lq_a = imfrombytes(self.file_client.get(lq_a_path, 'lq_a'), float32=True)
            img_lq_b = imfrombytes(self.file_client.get(lq_b_path, 'lq_b'), float32=True)

            # Validate images
            for name, img, path in [('GT', img_gt, gt_path), ('LQ_A', img_lq_a, lq_a_path), ('LQ_B', img_lq_b, lq_b_path)]:
                if img is None or img.size == 0:
                    raise ValueError(f"Invalid image {name}: {path}")
                if len(img.shape) != 3 or img.shape[2] != 3:
                    raise ValueError(f"Image {name} is not RGB: {path}")

            # Ensure proper size relationships
            h_gt, w_gt = img_gt.shape[:2]
            target_h_lq = h_gt // self.scale
            target_w_lq = w_gt // self.scale

            # Resize LQ images to maintain scale relationship
            img_lq_a = cv2.resize(img_lq_a, (target_w_lq, target_h_lq), interpolation=cv2.INTER_AREA)
            img_lq_b = cv2.resize(img_lq_b, (target_w_lq, target_h_lq), interpolation=cv2.INTER_AREA)

            # Training: crop fixed-size patches
            if self.phase == 'train' and self.gt_size is not None:
                lq_size = self.gt_size // self.scale
                
                # Ensure minimum size
                min_size = max(lq_size, 32)  # Minimum 32x32 LQ patch
                if img_lq_a.shape[0] < min_size or img_lq_a.shape[1] < min_size:
                    # Pad if too small
                    pad_h = max(0, min_size - img_lq_a.shape[0])
                    pad_w = max(0, min_size - img_lq_a.shape[1])
                    
                    for img in [img_lq_a, img_lq_b]:
                        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                    
                    img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h * self.scale, 0, pad_w * self.scale, cv2.BORDER_REFLECT_101)

                # Random crop
                if img_lq_a.shape[0] >= lq_size and img_lq_a.shape[1] >= lq_size:
                    top = random.randint(0, img_lq_a.shape[0] - lq_size)
                    left = random.randint(0, img_lq_a.shape[1] - lq_size)
                    
                    img_lq_a = img_lq_a[top:top + lq_size, left:left + lq_size, :]
                    img_lq_b = img_lq_b[top:top + lq_size, left:left + lq_size, :]
                    img_gt = img_gt[top*self.scale:(top + lq_size)*self.scale, 
                                  left*self.scale:(left + lq_size)*self.scale, :]

            # Apply enhanced augmentation
            img_gt, img_lq_a, img_lq_b = self.apply_additional_augmentation([img_gt, img_lq_a, img_lq_b])
            
            # Standard augmentation (flip, rotate)
            img_gt, img_lq_a, img_lq_b = augment([img_gt, img_lq_a, img_lq_b], self.use_flip, self.use_rot)

            # Convert to tensor
            img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
            img_lq_a = img2tensor(img_lq_a, bgr2rgb=True, float32=True)  
            img_lq_b = img2tensor(img_lq_b, bgr2rgb=True, float32=True)

            # Apply normalization if specified (but recommend keeping in [0,1])
            if self.mean is not None and self.std is not None:
                normalize(img_gt, self.mean, self.std, inplace=True)
                normalize(img_lq_a, self.mean, self.std, inplace=True)
                normalize(img_lq_b, self.mean, self.std, inplace=True)

            return {
                'gt': img_gt,
                'lq_a': img_lq_a,
                'lq_b': img_lq_b,
                'gt_path': gt_path,
                'lq_a_path': lq_a_path,
                'lq_b_path': lq_b_path
            }

        except Exception as e:
            print(f"[ERROR] Loading index {index}: {e}")
            # Return next valid sample
            return self.__getitem__((index + 1) % self.__len__())

    def __len__(self):
        return len(self.paths_gt)