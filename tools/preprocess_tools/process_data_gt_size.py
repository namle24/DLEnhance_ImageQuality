import os
import cv2

# Cấu hình thư mục
gt_dir = '/datasets/RealESRGAN_data/dataset/train/HR_sub'
lq_a_dir = '/datasets/RealESRGAN_data/dataset/train/LR_light_sub'
lq_b_dir = '/datasets/RealESRGAN_data/dataset/train/LR_moderate_sub'
gt_size = 128  # thay đổi theo cấu hình của bạn

img_names = sorted(os.listdir(gt_dir))

num_removed = 0

for name in img_names:
    gt_path = os.path.join(gt_dir, name)
    lq_a_path = os.path.join(lq_a_dir, name)
    lq_b_path = os.path.join(lq_b_dir, name)

    if not os.path.exists(gt_path) or not os.path.exists(lq_a_path) or not os.path.exists(lq_b_path):
        print(f"Bỏ qua {name}: thiếu ảnh.")
        continue

    gt = cv2.imread(gt_path)
    lq_a = cv2.imread(lq_a_path)
    lq_b = cv2.imread(lq_b_path)

    if gt is None or lq_a is None or lq_b is None:
        print(f"Ảnh lỗi: {name}")
        continue

    h1, w1 = gt.shape[:2]
    h2, w2 = lq_a.shape[:2]
    h3, w3 = lq_b.shape[:2]

    if min(h1, h2, h3) < gt_size or min(w1, w2, w3) < gt_size:
        print(f"Xóa {name} do kích thước nhỏ hơn {gt_size}")
        os.remove(gt_path)
        os.remove(lq_a_path)
        os.remove(lq_b_path)
        num_removed += 1

print(f"Đã xóa {num_removed} ảnh không đạt yêu cầu.")
