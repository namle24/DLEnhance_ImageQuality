import torch
from torch.utils.data import Dataset
import cv2
import os
import glob
import numpy as np
from PD_MAE_SR.utils.pd_mae_utils import compute_complexity_mask

class PDMAEDataset(Dataset):
    def __init__(self, root_dirs, patch_size=256, mae_patch_size=8, degrade_ratio=0.75):
        """
        Args:
            root_dirs: List of paths to pre-generated PD folders (PD_60, PD_70, etc.)
            patch_size: Training patch size.
            mae_patch_size: MAE patch size.
        """
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
            
        self.samples = []
        for root in root_dirs:
            lq_dir = os.path.join(root, 'PD_LQ')
            hr_dir = os.path.join(root, 'HR')
            
            # Find common filenames
            fnames = [os.path.basename(f) for f in glob.glob(os.path.join(lq_dir, "*.png"))]
            for f in fnames:
                self.samples.append({
                    'lq_path': os.path.join(lq_dir, f),
                    'hr_path': os.path.join(hr_dir, f),
                    'fname': f
                })
        
        self.patch_size = patch_size
        self.mae_patch_size = mae_patch_size
        self.degrade_ratio = degrade_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        lq_img = cv2.imread(sample['lq_path'])
        hr_img = cv2.imread(sample['hr_path'])
        
        if lq_img is None or hr_img is None:
            return self.__getitem__(np.random.randint(0, len(self)))

        h, w = hr_img.shape[:2]
        
        # 1. Random Crop to patch_size
        if h > self.patch_size or w > self.patch_size:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            lq_patch = lq_img[top:top+self.patch_size, left:left+self.patch_size]
            hr_patch = hr_img[top:top+self.patch_size, left:left+self.patch_size]
        else:
            lq_patch = cv2.resize(lq_img, (self.patch_size, self.patch_size))
            hr_patch = cv2.resize(hr_img, (self.patch_size, self.patch_size))

        # 2. Re-compute Mask from HR patch (Novelty: Complexity-based)
        mask_binary = compute_complexity_mask(hr_patch, degrade_ratio=self.degrade_ratio, patch_size=self.mae_patch_size)
        
        # 3. Downsample mask to MAE patch grid (e.g., 256/8 = 32x32)
        # mask_binary is 256x256, we need 32x32
        mask_patch_grid = cv2.resize(mask_binary.astype(np.uint8), 
                                     (self.patch_size // self.mae_patch_size, 
                                      self.patch_size // self.mae_patch_size), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # Convert to RGB and Tensors
        lq_rgb = cv2.cvtColor(lq_patch, cv2.COLOR_BGR2RGB)
        hr_rgb = cv2.cvtColor(hr_patch, cv2.COLOR_BGR2RGB)
        
        lq_tensor = torch.from_numpy(lq_rgb).permute(2, 0, 1).float() / 255.0
        hr_tensor = torch.from_numpy(hr_rgb).permute(2, 0, 1).float() / 255.0
        
        # Flatten the patch-level mask
        mask_indices = mask_patch_grid.astype(np.int64).flatten()
        mask_indices = torch.from_numpy(mask_indices)
        
        return {
            'lq': lq_tensor,
            'hr': hr_tensor,
            'mask_indices': mask_indices,
            'filename': sample['fname']
        }
