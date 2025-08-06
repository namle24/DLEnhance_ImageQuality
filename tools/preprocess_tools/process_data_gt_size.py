import os
import cv2

# Danh sách folder cần kiểm tra
folders = {
    "HR_sub": "/datasets/RealESRGAN_data/dataset/train/HR_sub",
    "LR_light_sub": "/datasets/RealESRGAN_data/dataset/train/LR_light_sub",
    "LR_moderate_sub": "/datasets/RealESRGAN_data/dataset/train/LR_moderate_sub"
}

# Định dạng ảnh hợp lệ
valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')

print("=== BẮT ĐẦU KIỂM TRA DATASET ===")
for name, folder in folders.items():
    print(f"\n--- Kiểm tra thư mục: {name} ({folder}) ---")
    error_count = 0
    total_count = 0

    for fname in os.listdir(folder):
        total_count += 1
        fpath = os.path.join(folder, fname)

        # Bỏ qua file không phải ảnh
        if not fname.lower().endswith(valid_ext):
            print(f"Không phải file ảnh: {fpath}")
            error_count += 1
            continue

        # Thử đọc ảnh
        img = cv2.imread(fpath)
        if img is None:
            print(f"Không đọc được ảnh: {fpath}")
            error_count += 1

    print(f"Tổng số ảnh: {total_count}, Lỗi: {error_count}")

print("\n=== KIỂM TRA HOÀN TẤT ===")
