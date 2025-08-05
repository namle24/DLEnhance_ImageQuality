import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(pred, gt):
    pred_np = pred.mul(255).byte().cpu().numpy().transpose(0, 2, 3, 1)
    gt_np = gt.mul(255).byte().cpu().numpy().transpose(0, 2, 3, 1)
    return peak_signal_noise_ratio(gt_np, pred_np, data_range=255)

def calculate_ssim(pred, gt):
    pred_np = pred.mul(255).byte().cpu().numpy().transpose(0, 2, 3, 1)
    gt_np = gt.mul(255).byte().cpu().numpy().transpose(0, 2, 3, 1)
    return structural_similarity(gt_np, pred_np, multichannel=True, data_range=255)