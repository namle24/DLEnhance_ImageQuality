import os
import argparse
from PIL import Image, ImageDraw

def get_safe_crop_box(img_size, target_box, crop_size=150):
    w, h = img_size
    if target_box:
        x1, y1, x2, y2 = target_box
        # If the target box is valid and fits inside the image
        if x2 - x1 >= crop_size and y2 - y1 >= crop_size:
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, w - crop_size))
            y1 = max(0, min(y1, h - crop_size))
            return (x1, y1, x1 + crop_size, y1 + crop_size)
    
    # Fallback to center crop
    x1 = max(0, w // 2 - crop_size // 2)
    y1 = max(0, h // 2 - crop_size // 2)
    return (x1, y1, x1 + crop_size, y1 + crop_size)

def crop_and_save_dataset(dataset_cfg, base_results_dir, base_datasets_dir, output_dir, scale=4, crop_size=150):
    name = dataset_cfg['name']
    img_filename = dataset_cfg['img_filename']
    target_box = dataset_cfg['crop_box']
    
    print(f"\n--- Processing benchmark: {name} (Image: {img_filename}) ---")
    
    # Resolve paths
    gt_path = os.path.join(base_datasets_dir, dataset_cfg['gt_rel'], img_filename)
    lq_path = os.path.join(base_datasets_dir, dataset_cfg['lq_rel'], img_filename)
    
    # In case LQ has a slightly different extension or path structure
    if not os.path.exists(lq_path):
        # Try lowercase or common patterns if needed
        pass
        
    baseline_filename = img_filename.replace('.png', f"_test_DRCT_baseline_x4.png")
    ours_filename = img_filename.replace('.png', f"_test_PD_MAE_DRCT_x4.png")
    
    baseline_path = os.path.join(base_results_dir, "test_DRCT_baseline_x4", "visualization", dataset_cfg['viz_sub'], baseline_filename)
    ours_path = os.path.join(base_results_dir, "test_PD_MAE_DRCT_x4", "visualization", dataset_cfg['viz_sub'], ours_filename)
    
    # Validate existences
    missing_files = []
    for label, path in [("GT", gt_path), ("LQ", lq_path), ("Baseline", baseline_path), ("Ours", ours_path)]:
        if not os.path.exists(path):
            missing_files.append((label, path))
            
    if missing_files:
        print(f"Error: Missing files for dataset {name}:")
        for label, path in missing_files:
            print(f"  - {label} not found at: {path}")
        print("Skipping this dataset. Please make sure paths are correct on the server.")
        return False
        
    # Load images
    img_gt = Image.open(gt_path).convert('RGB')
    img_lq = Image.open(lq_path).convert('RGB')
    img_base = Image.open(baseline_path).convert('RGB')
    img_ours = Image.open(ours_path).convert('RGB')
    
    # Get safe crop box
    crop_box = get_safe_crop_box(img_gt.size, target_box, crop_size=crop_size)
    x1, y1, x2, y2 = crop_box
    print(f"Applying crop box (x1, y1, x2, y2): {crop_box} on GT size: {img_gt.size}")
    
    # Crop HR-sized images
    crop_gt = img_gt.crop(crop_box)
    crop_base = img_base.crop(crop_box)
    crop_ours = img_ours.crop(crop_box)
    
    # Crop LR image (scale down coordinates)
    x1_lr, y1_lr = int(x1 / scale), int(y1 / scale)
    x2_lr, y2_lr = int(x2 / scale), int(y2 / scale)
    crop_lr = img_lq.crop((x1_lr, y1_lr, x2_lr, y2_lr))
    crop_lr_resized = crop_lr.resize((crop_size, crop_size), Image.Resampling.BICUBIC)
    
    # Save outputs with clean naming convention
    os.makedirs(output_dir, exist_ok=True)
    
    # Use hyphens instead of underscores to avoid LaTeX compilation issues with filenames
    name_clean = name.lower().replace('_', '-')
    out_gt = os.path.join(output_dir, f"{name_clean}-gt.png")
    out_lq = os.path.join(output_dir, f"{name_clean}-lq.png")
    out_base = os.path.join(output_dir, f"{name_clean}-baseline.png")
    out_ours = os.path.join(output_dir, f"{name_clean}-ours.png")
    
    crop_gt.save(out_gt)
    crop_lr_resized.save(out_lq)
    crop_base.save(out_base)
    crop_ours.save(out_ours)
    
    print(f"Successfully saved crops to {output_dir}:")
    print(f"  - GT: {os.path.basename(out_gt)}")
    print(f"  - LQ: {os.path.basename(out_lq)}")
    print(f"  - Baseline: {os.path.basename(out_base)}")
    print(f"  - Ours: {os.path.basename(out_ours)}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop benchmark images for LaTeX report.")
    parser.add_argument("--results_dir", type=str, default="/data/home/namlh/DLEnhance_ImageQuality/results/pd_mae",
                        help="Path to results/pd_mae containing visualization directories")
    parser.add_argument("--datasets_dir", type=str, default="/data/home/namlh/data/benchmark_datasets",
                        help="Path to benchmark_datasets directory")
    parser.add_argument("--output_dir", type=str, default="/data/home/namlh/DLEnhance_ImageQuality/results/pd_mae",
                        help="Path to save cropped images (should be same as referenced in latex pd_mae/)")
    parser.add_argument("--crop_size", type=int, default=160, help="Size of cropped square patch")
    args = parser.parse_args()

    # Define configuration for each benchmark dataset
    datasets_config = [
        {
            "name": "Set5",
            "img_filename": "butterfly.png",
            "gt_rel": "Set5/Set5/GTmod12",
            "lq_rel": "Set5/Set5/LRbicx4",
            "viz_sub": "Set5",
            "crop_box": (260, 220, 420, 380) # target detailed wing pattern
        },
        {
            "name": "Set14",
            "img_filename": "zebra.png",
            "gt_rel": "Set14/Set14/GTmod12",
            "lq_rel": "Set14/Set14/LRbicx4",
            "viz_sub": "Set14",
            "crop_box": (280, 140, 440, 300) # target zebra stripes
        },
        {
            "name": "Urban100",
            "img_filename": "img_002.png",
            "gt_rel": "urban100/urban100",
            "lq_rel": "urban100/urban100_LQ",
            "viz_sub": "Urban100",
            "crop_box": (400, 300, 560, 460) # target window lattice
        },
        {
            "name": "BSDS100",
            "img_filename": "108070.png",
            "gt_rel": "BSDS100/BSDS100",
            "lq_rel": "BSDS100/BSD100_LQ",
            "viz_sub": "BSDS100",
            "crop_box": (180, 80, 340, 240) # target tiger fur texture
        },
        {
            "name": "RealSR_Canon",
            "img_filename": "Canon_004.png",
            "gt_rel": "RealSR(V3)/Canon/HR",
            "lq_rel": "RealSR(V3)/Canon/LR",
            "viz_sub": "RealSR_Canon",
            "crop_box": (800, 600, 960, 760) # target high-frequency details
        },
        {
            "name": "RealSR_Nikon",
            "img_filename": "Nikon_050.png",
            "gt_rel": "RealSR(V3)/Nikon/HR",
            "lq_rel": "RealSR(V3)/Nikon/LR",
            "viz_sub": "RealSR_Nikon",
            "crop_box": (1000, 800, 1160, 960) # target noisy fabric or background details
        }
    ]

    print("=== STARTING CROP PROCESS FOR ALL BENCHMARKS ===")
    print(f"Results Dir: {args.results_dir}")
    print(f"Datasets Dir: {args.datasets_dir}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Crop Size: {args.crop_size}x{args.crop_size}")
    
    success_count = 0
    for dataset_cfg in datasets_config:
        success = crop_and_save_dataset(
            dataset_cfg=dataset_cfg,
            base_results_dir=args.results_dir,
            base_datasets_dir=args.datasets_dir,
            output_dir=args.output_dir,
            scale=4,
            crop_size=args.crop_size
        )
        if success:
            success_count += 1
            
    print(f"\n=== PROCESS FINISHED: Cropped {success_count}/{len(datasets_config)} datasets successfully ===")
