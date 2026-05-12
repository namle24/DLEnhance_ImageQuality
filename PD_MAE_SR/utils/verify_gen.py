import cv2
import numpy as np
import os

lq_path = 'PD_MAE_SR/test_dataset/PD_LQ/00003_p256_l80_7498.png'
hr_path = 'PD_MAE_SR/test_dataset/HR/00003_p256_l80_7498.png'
mask_path = 'PD_MAE_SR/test_dataset/mask/00003_p256_l80_7498.png'

lq = cv2.imread(lq_path)
hr = cv2.imread(hr_path)
mask = cv2.imread(mask_path)

if mask.ndim == 2:
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
else:
    mask_bgr = mask

combined = np.hstack((hr, mask_bgr, lq))
output_path = 'PD_MAE_SR/results/foundation/pd_mae_sample_verification.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cv2.imwrite(output_path, combined)
print(f"Verification image saved to {output_path}")
