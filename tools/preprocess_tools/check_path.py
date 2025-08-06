import os
import cv2

def check_image(path):
    if not os.path.exists(path):
        return False, 'not_exist'
    img = cv2.imread(path)
    if img is None:
        return False, 'unreadable'
    h, w = img.shape[:2]
    if h < 120 or w < 120:
        return False, f'too_small: {w}x{h}'
    return True, 'ok'

root_gt = '/datasets/RealESRGAN_data/dataset/train/HR_sub'
root_lq_a = '/datasets/RealESRGAN_data/dataset/train/LR_light_sub'
root_lq_b = '/datasets/RealESRGAN_data/dataset/train/LR_moderate_sub'

errors = []

for dirpath, _, filenames in os.walk(root_gt):
    for fname in filenames:
        rel_path = os.path.relpath(os.path.join(dirpath, fname), root_gt)
        path_gt = os.path.join(root_gt, rel_path)
        path_lq_a = os.path.join(root_lq_a, rel_path)
        path_lq_b = os.path.join(root_lq_b, rel_path)

        for p, tag in [(path_gt, 'GT'), (path_lq_a, 'LQ_A'), (path_lq_b, 'LQ_B')]:
            ok, status = check_image(p)
            if not ok:
                errors.append((tag, p, status))

print(f"Found {len(errors)} problematic images.")
for e in errors:
    print(e)
