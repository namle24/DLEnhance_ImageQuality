import torch
from torch.utils.data import Dataset
import cv2
import os
import glob
import numpy as np

class PDMAEDataset(Dataset):
    def __init__(self, root_dirs, patch_size=256, mae_patch_size=8):
        """
        Args:
            root_dirs: String or List of strings containing 'PD_LQ', 'HR', and 'mask' subdirectories.
        """
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
            
        self.samples = []
        for root in root_dirs:
            lq_dir = os.path.join(root, 'PD_LQ')
            hr_dir = os.path.join(root, 'HR')
            mask_dir = os.path.join(root, 'mask')
            
            # Find common filenames (assuming filenames are same across subdirs)
            fnames = [os.path.basename(f) for f in glob.glob(os.path.join(lq_dir, "*.png"))]
            for f in fnames:
                self.samples.append({
                    'lq_path': os.path.join(lq_dir, f),
                    'hr_path': os.path.join(hr_dir, f),
                    'mask_path': os.path.join(mask_dir, f),
                    'fname': f
                })
        
        self.patch_size = patch_size
        self.mae_patch_size = mae_patch_size
        self.num_mae_patches = (patch_size // mae_patch_size) ** 2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fname = sample['fname']
        
        lq_path = sample['lq_path']
        hr_path = sample['hr_path']
        mask_path = sample['mask_path']
        
        lq_img = cv2.imread(lq_path)
        hr_img = cv2.imread(hr_path)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Random Crop if image is larger than patch_size
        h, w = hr_img.shape[:2]
        if h > self.patch_size or w > self.patch_size:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            lq_img = lq_img[top:top+self.patch_size, left:left+self.patch_size]
            hr_img = hr_img[top:top+self.patch_size, left:left+self.patch_size]
            mask_img = mask_img[top:top+self.patch_size, left:left+self.patch_size]
        elif h < self.patch_size or w < self.patch_size:
            # Padding or resizing if smaller (should not happen with 480x480)
            lq_img = cv2.resize(lq_img, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
            hr_img = cv2.resize(hr_img, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
            mask_img = cv2.resize(mask_img, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)

        lq_img = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        lq_tensor = torch.from_numpy(lq_img).permute(2, 0, 1).float() / 255.0
        hr_tensor = torch.from_numpy(hr_img).permute(2, 0, 1).float() / 255.0
        
        # The mask in folder is full resolution. 
        # MAE needs a patch-level binary mask [L] where 1 is masked (degraded)
        # We can downsample the mask to mae_patch grid
        mask_patch_grid = cv2.resize(mask_img, 
                                     (self.patch_size // self.mae_patch_size, 
                                      self.patch_size // self.mae_patch_size), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # Threshold to binary (mask folder usually has 0 and 255)
        mask_indices = (mask_patch_grid > 127).astype(np.int64).flatten()
        mask_indices = torch.from_numpy(mask_indices)
        
        return {
            'lq': lq_tensor,
            'hr': hr_tensor,
            'mask_indices': mask_indices,
            'filename': fname
        }
