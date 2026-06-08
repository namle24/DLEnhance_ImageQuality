import os
from PIL import Image, ImageDraw

def crop_and_bbox(img_dir, set_name, lr_name, base_name, ours_name, gt_name, crop_box, scale=4):
    """
    img_dir: Directory containing the images
    set_name: Prefix name for output files (e.g. 'Canon_004')
    lr_name: Filename of LR input
    base_name: Filename of Baseline result
    ours_name: Filename of PD-MAE result
    gt_name: Filename of GT reference
    crop_box: (x1, y1, x2, y2) coordinates on the GT/HR image (which is 4x larger than LR)
    scale: SR scale factor
    """
    print(f"--- Processing set: {set_name} ---")
    
    # Paths
    lr_path = os.path.join(img_dir, lr_name)
    base_path = os.path.join(img_dir, base_name)
    ours_path = os.path.join(img_dir, ours_name)
    gt_path = os.path.join(img_dir, gt_name)
    
    # Load images
    img_lr = Image.open(lr_path).convert('RGB')
    img_base = Image.open(base_path).convert('RGB')
    img_ours = Image.open(ours_path).convert('RGB')
    img_gt = Image.open(gt_path).convert('RGB')
    
    # Get dimensions
    w_gt, h_gt = img_gt.size
    print(f"GT dimensions: {w_gt}x{h_gt}")
    
    x1, y1, x2, y2 = crop_box
    # Ensure coordinates are within bounds
    x1, x2 = max(0, x1), min(w_gt, x2)
    y1, y2 = max(0, y1), min(h_gt, y2)
    
    # Crop HR-sized images (Baseline, Ours, GT)
    crop_base = img_base.crop((x1, y1, x2, y2))
    crop_ours = img_ours.crop((x1, y1, x2, y2))
    crop_gt = img_gt.crop((x1, y1, x2, y2))
    
    # Crop LR image (which is 4x smaller, so scale coordinates down)
    x1_lr = int(x1 / scale)
    y1_lr = int(y1 / scale)
    x2_lr = int(x2 / scale)
    y2_lr = int(y2 / scale)
    crop_lr = img_lr.crop((x1_lr, y1_lr, x2_lr, y2_lr))
    # Resize LR crop to match HR crop size for visual comparison
    crop_lr_resized = crop_lr.resize((x2 - x1, y2 - y1), Image.Resampling.BICUBIC)
    
    # Draw red bounding box on the full GT image
    img_gt_bbox = img_gt.copy()
    draw = ImageDraw.Draw(img_gt_bbox)
    # Draw line with thickness
    linewidth = max(2, int(min(w_gt, h_gt) * 0.005))
    draw.rectangle([x1, y1, x2, y2], outline="red", width=linewidth)
    
    # Save results
    gt_bbox_name = f"{set_name}_bbox.png"
    lr_crop_name = f"{lr_name.replace('.png', '')}_crop.png"
    base_crop_name = f"{base_name.replace('.png', '')}_crop.png"
    ours_crop_name = f"{ours_name.replace('.png', '')}_crop.png"
    gt_crop_name = f"{gt_name.replace('.png', '')}_crop.png"
    
    img_gt_bbox.save(os.path.join(img_dir, gt_bbox_name))
    crop_lr_resized.save(os.path.join(img_dir, lr_crop_name))
    crop_base.save(os.path.join(img_dir, base_crop_name))
    crop_ours.save(os.path.join(img_dir, ours_crop_name))
    crop_gt.save(os.path.join(img_dir, gt_crop_name))
    
    print(f"Saved full image with bbox to: {gt_bbox_name}")
    print(f"Saved LQ crop to: {lr_crop_name}")
    print(f"Saved Baseline crop to: {base_crop_name}")
    print(f"Saved PD-MAE crop to: {ours_crop_name}")
    print(f"Saved GT crop to: {gt_crop_name}")

if __name__ == "__main__":
    # Base directory for the images
    img_dir = r"D:\Projects\DLEnhance_ImageQuality\results\pd_mae"
    
    # -------------------------------------------------------------
    # 1. Processing Set 1: Canon_004
    # Let's specify crop box. (x1, y1, x2, y2) on HR image.
    # If the image is e.g. 1000x1000, we crop a 128x128 or 256x256 region.
    # Let's target a text region or detailed pattern.
    # The user can adjust these coordinates as needed.
    # Let's use coordinates near the center as a default:
    # -------------------------------------------------------------
    crop_box_canon = (300, 300, 500, 500) # x1, y1, x2, y2 (Adjustable)
    crop_and_bbox(
        img_dir=img_dir,
        set_name="Canon_004",
        lr_name="InputLR_Canon_004.png",
        base_name="Baseline_Canon_004_500000.png",
        ours_name="PD_MAE_Canon_004_500000.png",
        gt_name="Canon_004.png",
        crop_box=crop_box_canon,
        scale=4
    )
    
    # -------------------------------------------------------------
    # 2. Processing Set 2: Set5 butterfly
    # butterfly.png is typically 256x256 or 512x512.
    # Let's target the wing textures near (200, 100, 350, 250) or similar.
    # -------------------------------------------------------------
    crop_box_bf = (150, 150, 278, 278) # 128x128 region (Adjustable)
    crop_and_bbox(
        img_dir=img_dir,
        set_name="butterfly",
        lr_name="InputLR_butterfly.png",
        base_name="Baseline_butterfly_500000.png",
        ours_name="PD_MAE_butterfly_500000.png",
        gt_name="butterfly.png",
        crop_box=crop_box_bf,
        scale=4
    )
    
    # -------------------------------------------------------------
    # 3. Processing Set 3: Nikon_018
    # Let's specify crop box for Nikon camera noise set.
    # -------------------------------------------------------------
    crop_box_nikon = (400, 400, 600, 600) # x1, y1, x2, y2 (Adjustable)
    crop_and_bbox(
        img_dir=img_dir,
        set_name="Nikon_018",
        lr_name="InputLR_Nikon_018.png",
        base_name="Baseline_Nikon_018_500000.png",
        ours_name="PD_MAE_Nikon_018_500000.png",
        gt_name="Nikon_018.png",
        crop_box=crop_box_nikon,
        scale=4
    )
