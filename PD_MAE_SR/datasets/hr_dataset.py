import torch
from torch.utils.data import Dataset
import cv2
import os
import glob
import numpy as np

class HROnlyDataset(Dataset):
    def __init__(self, root_dir, patch_size=256, mae_patch_size=8, degrade_ratio=0.75):
        """
        Args:
            root_dir: Path to the HR images folder (e.g., ~/data/dataset/train/HR_sub)
            patch_size: Training patch size.
            mae_patch_size: MAE patch size.
            degrade_ratio: Ratio of patches to mask (default: 0.75)
        """
        self.root_dir = os.path.expanduser(root_dir)
        
        # Support loading standard image files
        self.samples = []
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        for ext in extensions:
            self.samples.extend(glob.glob(os.path.join(self.root_dir, ext)))
            
        self.samples = sorted(self.samples)
        
        self.patch_size = patch_size
        self.mae_patch_size = mae_patch_size
        self.degrade_ratio = degrade_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        hr_img = cv2.imread(img_path)
        
        if hr_img is None:
            # Fallback to a random sample if reading fails
            return self.__getitem__(np.random.randint(0, len(self)))

        h, w = hr_img.shape[:2]
        
        # 1. Random Crop to patch_size
        if h > self.patch_size or w > self.patch_size:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            hr_patch = hr_img[top:top+self.patch_size, left:left+self.patch_size]
        else:
            hr_patch = cv2.resize(hr_img, (self.patch_size, self.patch_size))

        # 2. Random Grid Mask 75% (8x8 patches)
        grid_h = self.patch_size // self.mae_patch_size
        grid_w = self.patch_size // self.mae_patch_size
        num_patches = grid_h * grid_w
        num_masked = int(num_patches * self.degrade_ratio)
        
        # Generate exact random mask
        mask_flat = np.zeros(num_patches, dtype=np.float32)
        masked_indices = np.random.choice(num_patches, size=num_masked, replace=False)
        mask_flat[masked_indices] = 1.0
        
        mask_patch_grid = mask_flat.reshape(grid_h, grid_w)
        
        # Upscale mask back to patch_size x patch_size to zero out pixels
        mask_binary = cv2.resize(mask_patch_grid, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
        
        # Zero out the masked patches to make the LQ input (black patches)
        lq_patch = hr_patch.copy()
        lq_patch[mask_binary == 1] = 0
        
        # Convert to RGB and Tensors
        lq_rgb = cv2.cvtColor(lq_patch, cv2.COLOR_BGR2RGB)
        hr_rgb = cv2.cvtColor(hr_patch, cv2.COLOR_BGR2RGB)
        
        lq_tensor = torch.from_numpy(lq_rgb).permute(2, 0, 1).float() / 255.0
        hr_tensor = torch.from_numpy(hr_rgb).permute(2, 0, 1).float() / 255.0
        
        # Flatten the patch-level mask indices (1 = masked, 0 = keep)
        mask_indices = torch.from_numpy(mask_flat).long()
        
        return {
            'lq': lq_tensor,
            'hr': hr_tensor,
            'mask_indices': mask_indices,
            'filename': os.path.basename(img_path)
        }
