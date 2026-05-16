import torch
from torch.utils.data import Dataset
import cv2
import os
import glob
import numpy as np
import random
from PD_MAE_SR.utils.pd_mae_utils import compute_complexity_mask

class PDMAEDataset(Dataset):
    def __init__(self, hr_dir, lq_dirs, patch_size=256, mae_patch_size=8, degrade_ratio=0.75):
        """
        Args:
            hr_dir: Path to HR patches folder.
            lq_dirs: List of paths to corresponding LR patches folders (60, 70, 80, 90).
            patch_size: Size of the patch to feed the model.
            mae_patch_size: Size of MAE patches.
            degrade_ratio: Percentage of image to degrade (default 75%).
        """
        if isinstance(lq_dirs, str):
            lq_dirs = [lq_dirs]
            
        self.hr_dir = hr_dir
        self.lq_dirs = lq_dirs
        self.patch_size = patch_size
        self.mae_patch_size = mae_patch_size
        self.degrade_ratio = degrade_ratio
        
        # Get common filenames
        self.filenames = [os.path.basename(f) for f in glob.glob(os.path.join(hr_dir, "*.png"))]
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        hr_path = os.path.join(self.hr_dir, fname)
        
        # Randomly pick one degradation level
        lq_root = random.choice(self.lq_dirs)
        lq_path = os.path.join(lq_root, fname)
        
        hr_img = cv2.imread(hr_path)
        lq_img = cv2.imread(lq_path)
        
        if hr_img is None or lq_img is None:
            # Fallback for missing files
            return self.__getitem__(random.randint(0, len(self) - 1))

        h, w = hr_img.shape[:2]
        
        # Ensure LQ matches HR size (for standard x4 SR)
        if lq_img.shape != hr_img.shape:
            lq_img = cv2.resize(lq_img, (w, h), interpolation=cv2.INTER_NEAREST)

        # 1. Random Crop if needed
        if h > self.patch_size or w > self.patch_size:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            hr_patch = hr_img[top:top+self.patch_size, left:left+self.patch_size]
            lq_patch = lq_img[top:top+self.patch_size, left:left+self.patch_size]
        else:
            hr_patch = cv2.resize(hr_img, (self.patch_size, self.patch_size))
            lq_patch = cv2.resize(lq_img, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)

        # 2. Compute Complexity Mask (Dynamic)
        mask_binary = compute_complexity_mask(hr_patch, degrade_ratio=self.degrade_ratio, patch_size=self.mae_patch_size)
        
        # 3. Blend with Gaussian Feathering
        mask_f = mask_binary.astype(np.float32)
        mask_soft = cv2.GaussianBlur(mask_f, (15, 15), 3)
        mask_soft = np.expand_dims(mask_soft, axis=-1)
        
        pd_lq = (hr_patch.astype(np.float32) * (1.0 - mask_soft) + 
                 lq_patch.astype(np.float32) * mask_soft).astype(np.uint8)

        # Convert to RGB and Tensors
        pd_lq_rgb = cv2.cvtColor(pd_lq, cv2.COLOR_BGR2RGB)
        hr_patch_rgb = cv2.cvtColor(hr_patch, cv2.COLOR_BGR2RGB)
        
        lq_tensor = torch.from_numpy(pd_lq_rgb).permute(2, 0, 1).float() / 255.0
        hr_tensor = torch.from_numpy(hr_patch_rgb).permute(2, 0, 1).float() / 255.0
        
        mask_indices = mask_binary.astype(np.int64).flatten()
        mask_indices = torch.from_numpy(mask_indices)
        
        return {
            'lq': lq_tensor,
            'hr': hr_tensor,
            'mask_indices': mask_indices,
            'filename': fname
        }
