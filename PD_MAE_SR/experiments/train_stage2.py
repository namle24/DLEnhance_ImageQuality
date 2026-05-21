import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import logging
from tqdm import tqdm
import sys
import time
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from PD_MAE_SR.archs.mae_arch import PDMAE
from PD_MAE_SR.datasets.hr_dataset import HROnlyDataset

def patchify(imgs, p=8):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 * 3)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x

def unpatchify(x, p=8):
    """
    x: (N, L, patch_size**2 * 3)
    imgs: (N, 3, H, W)
    """
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
    return imgs

def train():
    parser = argparse.ArgumentParser(description='PD-MAE Stage 2 HR Structure Fine-Tuning')
    parser.add_argument('--data_root', type=str, required=True, help='Path to HR_sub folder containing clean HR images')
    parser.add_argument('--output_dir', type=str, default='PD_MAE_SR/checkpoints/stage2', help='Output directory')
    parser.add_argument('--pretrain', type=str, default=None, help='Path to Stage 1 checkpoint (e.g. pd_mae_s1_iter200000.pth)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (Stage 2 uses 1e-5)')
    parser.add_argument('--iters', type=int, default=100000, help='Total iterations (Stage 2 uses 100000)')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--mae_patch_size', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=5000)
    parser.add_argument('--val_freq', type=int, default=10000)
    parser.add_argument('--resume', type=str, default=None, help='Path to Stage 2 checkpoint to resume from')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Logging setup
    log_file = os.path.join(args.output_dir, 'train.log')
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project="PD-MAE-SR", config=args, name="Stage2-FineTune-HR")
        except ImportError:
            logging.warning("wandb is not installed. Running without wandb logging.")
            args.use_wandb = False
    
    # 1. Setup Data (Clean HR with 75% Random Grid Mask)
    dataset = HROnlyDataset(args.data_root, 
                            patch_size=args.patch_size, 
                            mae_patch_size=args.mae_patch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # 2. Setup Model
    model = PDMAE(img_size=args.patch_size, patch_size=args.mae_patch_size).cuda()
    
    # 3. Load pre-trained weights from Stage 1 if provided (Only model weights, reset optimizer/scheduler)
    if args.pretrain:
        if os.path.exists(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            # Load with strict matching
            model.load_state_dict(state_dict, strict=True)
            logging.info(f"Successfully loaded Stage 1 pre-trained model weights from {args.pretrain} (Optimizer/Scheduler reset).")
        else:
            raise FileNotFoundError(f"Stage 1 pre-trained checkpoint not found at: {args.pretrain}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    # Scheduler: Cosine Decay down to 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)
    
    criterion = nn.MSELoss()
    
    current_iter = 0
    
    # Resume Stage 2 Logic (if Stage 2 gets interrupted and resumed)
    if args.resume:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            current_iter = checkpoint['iter']
            logging.info(f"Resumed Stage 2 training from iter {current_iter}")
        else:
            raise FileNotFoundError(f"Stage 2 resume checkpoint not found at: {args.resume}")
    
    logging.info(f"Starting Stage 2 HR Structure Fine-Tuning for {args.iters} iterations...")
    
    pbar = tqdm(total=args.iters, initial=current_iter)
    
    while current_iter < args.iters:
        for batch in dataloader:
            if current_iter >= args.iters:
                break
                
            lq = batch['lq'].cuda()
            hr = batch['hr'].cuda()
            mask_indices = batch['mask_indices'].cuda() # [B, L]
            
            # Forward
            pred = model(lq) # [B, L, p*p*3]
            
            # Target patchification
            target = patchify(hr, p=args.mae_patch_size) # [B, L, p*p*3]
            
            # Loss only on masked patches (mask_indices == 1)
            mask_bool = mask_indices.bool()
            
            # Compute loss only where mask_bool is True
            if mask_bool.any():
                loss = criterion(pred[mask_bool], target[mask_bool])
            else:
                loss = torch.tensor(0.0).cuda()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            current_iter += 1
            pbar.update(1)
            pbar.set_description(f"Loss: {loss.item():.6f} LR: {scheduler.get_last_lr()[0]:.2e}")
            
            # Logging to file every 100 iters
            if current_iter % 100 == 0:
                logging.info(f"Iter {current_iter}/{args.iters} - Loss: {loss.item():.6f} - LR: {scheduler.get_last_lr()[0]:.2e}")
                if args.use_wandb:
                    wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=current_iter)
            
            # Save Checkpoint
            if current_iter % args.save_freq == 0:
                save_path = os.path.join(args.output_dir, f"pd_mae_s2_iter{current_iter}.pth")
                torch.save({
                    'iter': current_iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }, save_path)
                logging.info(f"Saved checkpoint: {save_path}")
 
            # Simple Validation / Visualization
            if current_iter % args.val_freq == 0 or current_iter == 100:
                model.eval()
                with torch.no_grad():
                    # Just take the last batch for visualization
                    recon = unpatchify(pred, p=args.mae_patch_size)
                    # Clamp and convert for saving
                    recon_img = (recon[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype('uint8')
                    hr_img = (hr[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype('uint8')
                    lq_img = (lq[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype('uint8')
                    
                    # Create comparison (Input lq / Predicted recon / Ground-Truth hr)
                    comparison = np.hstack((lq_img, recon_img, hr_img))
                    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
                    vis_path = os.path.join(args.output_dir, f"vis_iter{current_iter}.png")
                    cv2.imwrite(vis_path, comparison_bgr)
                    logging.info(f"Saved visualization: {vis_path}")
                    if args.use_wandb:
                        wandb.log({"reconstruction_vis": wandb.Image(comparison)}, step=current_iter)
                model.train()
                
    pbar.close()
    logging.info("Stage 2 Fine-Tuning complete.")
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    train()
