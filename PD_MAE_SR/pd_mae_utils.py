import cv2
import numpy as np
import os

def compute_complexity_mask(image, degrade_ratio=0.75, patch_size=8):
    """
    Computes a mask for region selection based on gradient complexity.
    Args:
        image (numpy array): Input HR image (H, W, C) in BGR.
        degrade_ratio (float): Ratio of patches to be degraded (masked).
        patch_size (int): Size of the patches for the MAE encoder.
    Returns:
        mask (numpy array): Binary mask (H, W) where 1 means "to be degraded".
        complexity_map (numpy array): Normalized complexity map.
    """
    h, w, _ = image.shape
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Compute gradients using Sobel
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 3. Magnitude of gradient
    complexity = np.sqrt(grad_x**2 + grad_y**2)
    
    # 4. Average complexity per patch (to match MAE patch structure)
    # Resize to patch grid size to compute per-patch threshold
    grid_h, grid_w = h // patch_size, w // patch_size
    
    # We use area interpolation to get average complexity per patch
    patch_complexity = cv2.resize(complexity, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    
    # 5. Determine threshold for top degrade_ratio%
    threshold = np.percentile(patch_complexity, (1 - degrade_ratio) * 100)
    
    # 6. Create patch-level mask
    patch_mask = (patch_complexity > threshold).astype(np.float32)
    
    # 7. Upsample mask to original resolution (nearest to keep blocky structure)
    mask = cv2.resize(patch_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Normalize complexity for visualization
    complexity_norm = cv2.normalize(complexity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return mask, complexity_norm

if __name__ == "__main__":
    # Test script
    img_path = "Real-ESRGAN/inputs/00003.png"
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        mask, complexity = compute_complexity_mask(img)
        
        # Visualize
        # Create a red overlay for degraded regions
        overlay = img.copy()
        overlay[mask == 1] = [0, 0, 255] # Mark as red
        
        combined = np.hstack((img, cv2.cvtColor(complexity, cv2.COLOR_GRAY2BGR), overlay))
        
        output_path = "scratch/pd_mae_foundation_test.png"
        os.makedirs("scratch", exist_ok=True)
        cv2.imwrite(output_path, combined)
        print(f"Visualization saved to {output_path}")
    else:
        print(f"File {img_path} not found.")
