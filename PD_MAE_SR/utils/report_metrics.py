import cv2
import numpy as np
import glob
import os

def analyze_mask_ratio(mask_dir):
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
    ratios = []
    for f in mask_files:
        mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        # Ratio of white pixels (degraded)
        ratio = np.sum(mask > 127) / mask.size
        ratios.append(ratio)
    return np.mean(ratios), ratios

def create_boundary_zoom(lq_path, mask_path, output_path):
    lq = cv2.imread(lq_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Find a boundary region (where mask changes from 0 to 255)
    # Simple way: find coordinates where mask is roughly 127 (after feathering)
    # Or just pick a fixed center area if we know it has mixed patches
    h, w = mask.shape
    
    # Let's create a zoom on a specific 64x64 area with high transition
    zoom_size = 64
    
    # Search for a good spot
    best_spot = (h//2, w//2)
    max_transition = 0
    for i in range(0, h - zoom_size, 16):
        for j in range(0, w - zoom_size, 16):
            sub = mask[i:i+zoom_size, j:j+zoom_size]
            transition = np.std(sub)
            if transition > max_transition:
                max_transition = transition
                best_spot = (i, j)
    
    y, x = best_spot
    crop_lq = lq[y:y+zoom_size, x:x+zoom_size]
    crop_mask = cv2.cvtColor(mask[y:y+zoom_size, x:x+zoom_size], cv2.COLOR_GRAY2BGR)
    
    # Upscale for better viewing
    upscale = 4
    crop_lq = cv2.resize(crop_lq, (zoom_size*upscale, zoom_size*upscale), interpolation=cv2.INTER_NEAREST)
    crop_mask = cv2.resize(crop_mask, (zoom_size*upscale, zoom_size*upscale), interpolation=cv2.INTER_NEAREST)
    
    combined = np.hstack((crop_mask, crop_lq))
    cv2.imwrite(output_path, combined)
    return output_path

if __name__ == "__main__":
    mask_dir = "PD_MAE_SR/test_dataset/mask"
    avg_ratio, all_ratios = analyze_mask_ratio(mask_dir)
    print(f"--- Statistics for {len(all_ratios)} images ---")
    print(f"Average Degraded Pixel Ratio: {avg_ratio*100:.2f}% (Target: ~75%)")
    
    # Pick the first image for boundary zoom
    lq_files = glob.glob("PD_MAE_SR/test_dataset/PD_LQ/*.png")
    if lq_files:
        lq_sample = lq_files[0]
        mask_sample = os.path.join(mask_dir, os.path.basename(lq_sample))
        zoom_path = "PD_MAE_SR/results/foundation/boundary_zoom_check.png"
        create_boundary_zoom(lq_sample, mask_sample, zoom_path)
        print(f"Boundary zoom check saved to {zoom_path}")
