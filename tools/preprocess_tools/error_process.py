import os

# Cấu hình thư mục
lq_a_root = "/datasets/RealESRGAN_data/dataset/train/LR_light_sub"
lq_b_root = "/datasets/RealESRGAN_data/dataset/train/LR_moderate_sub"
gt_root   = "/datasets/RealESRGAN_data/dataset/train/HR_sub"

# Đọc các dòng từ log lỗi
with open("tools/preprocess_tools/error_log.txt", "r") as f:
    lines = f.readlines()

deleted_count = 0
for line in lines:
    if "Ảnh LQ_A_cropped sau crop bị rỗng" in line:
        lq_a_path = line.strip().split(":")[-1].strip()
        filename = os.path.basename(lq_a_path)

        # Xoá ở 3 nơi nếu tồn tại
        for path in [
            os.path.join(lq_a_root, filename),
            os.path.join(lq_b_root, filename),
            os.path.join(gt_root, filename)
        ]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Đã xoá: {path}")
                deleted_count += 1

print(f"\nTổng số file đã xoá: {deleted_count}")
