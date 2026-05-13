import os
import cv2
import numpy as np
import random
import glob
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import logging
import sys

# Add path to import tools and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.degradation.RealisticDegradationGenerator_RealESRGAN import RealisticDegradationGenerator
from PD_MAE_SR.utils.pd_mae_utils import compute_complexity_mask

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def feather_mask(mask, kernel_size=15, sigma=3):
    """Apply Gaussian feathering to the mask boundary."""
    # Ensure mask is float32 for GaussianBlur
    mask_f = mask.astype(np.float32)
    feathered = cv2.GaussianBlur(mask_f, (kernel_size, kernel_size), sigma)
    return np.expand_dims(feathered, axis=-1)

def process_single_image(args):
    img_path, output_dir, scale, level, patch_size, save_mask, lq_dir = args
    
    try:
        hr_img = cv2.imread(img_path)
        if hr_img is None:
            return
        
        h, w = hr_img.shape[:2]
        
        # 0. Get LQ image if provided
        lq_full = None
        if lq_dir:
            fname = os.path.basename(img_path)
            lq_full_path = os.path.join(lq_dir, fname)
            lq_full = cv2.imread(lq_full_path)
            if lq_full is None:
                logging.warning(f"LQ image not found for {fname} in {lq_dir}")
                return
            if lq_full.shape != hr_img.shape:
                lq_full = cv2.resize(lq_full, (w, h), interpolation=cv2.INTER_NEAREST)

        # 1. Random crop to patch_size x patch_size
        if h < patch_size or w < patch_size:
            hr_img = cv2.resize(hr_img, (max(w, patch_size), max(h, patch_size)), interpolation=cv2.INTER_CUBIC)
            if lq_full is not None:
                lq_full = cv2.resize(lq_full, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            h, w = hr_img.shape[:2]
            
        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
        hr_patch = hr_img[top:top+patch_size, left:left+patch_size]
        
        # 2. Compute complexity mask (75% complexity)
        mask_binary = compute_complexity_mask(hr_patch, degrade_ratio=0.75, patch_size=16)
        
        # 3. Get Degraded version
        if lq_full is not None:
            lr_upsampled = lq_full[top:top+patch_size, left:left+patch_size]
        else:
            generator = RealisticDegradationGenerator()
            generator.set_level(level)
            lr_img = generator.apply_degradation(hr_patch, scale)
            lr_upsampled = cv2.resize(lr_img, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
        
        # 4. Blend with Gaussian Feathering
        mask_feathered = feather_mask(mask_binary, kernel_size=15, sigma=3)
        pd_lq = (hr_patch.astype(np.float32) * (1.0 - mask_feathered) + 
                 lr_upsampled.astype(np.float32) * mask_feathered).astype(np.uint8)
        
        # 5. Save results
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        suffix = f"_p{patch_size}_l{level}_{random.randint(1000, 9999)}"
        
        pd_lq_path = os.path.join(output_dir, "PD_LQ", f"{base_name}{suffix}.png")
        hr_path = os.path.join(output_dir, "HR", f"{base_name}{suffix}.png")
        
        os.makedirs(os.path.dirname(pd_lq_path), exist_ok=True)
        os.makedirs(os.path.dirname(hr_path), exist_ok=True)
        
        cv2.imwrite(pd_lq_path, pd_lq)
        cv2.imwrite(hr_path, hr_patch)
        
        if save_mask:
            mask_path = os.path.join(output_dir, "mask", f"{base_name}{suffix}.png")
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            cv2.imwrite(mask_path, (mask_binary * 255).astype(np.uint8))
            
    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate PD-MAE Dataset')
    parser.add_argument('--input', type=str, required=True, help='Input HR directory')
    parser.add_argument('--lq_dir', type=str, default=None, help='Optional: Pre-generated full-LQ directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--scale', type=float, default=0.25, help='Degradation scale (0.25 for x4)')
    parser.add_argument('--level', type=int, default=80, choices=[60, 70, 80, 90], help='Degradation level')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size for pretraining')
    parser.add_argument('--num_patches', type=int, default=1, help='Number of random patches per image')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--save_mask', action='store_true', help='Save masks for debugging')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logging.error(f"Input path {args.input} does not exist.")
        return
        
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(args.input, '**', ext), recursive=True))
        
    if not image_files:
        logging.error(f"No images found in {args.input}")
        return
        
    logging.info(f"Found {len(image_files)} images. Generating PD dataset...")
    
    # Duplicate files based on num_patches
    process_args = []
    for img_path in image_files:
        for _ in range(args.num_patches):
            process_args.append((img_path, args.output, args.scale, args.level, args.patch_size, args.save_mask, args.lq_dir))
            
    n_workers = args.workers if args.workers else max(1, cpu_count() - 1)
    
    with Pool(n_workers) as pool:
        list(tqdm(pool.imap(process_single_image, process_args), total=len(process_args)))
        
    logging.info("Dataset generation complete.")

if __name__ == '__main__':
    main()
