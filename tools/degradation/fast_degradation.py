import os
import cv2
import numpy as np

input_dir = r"D:\Download\archive\BSDS100\BSD100"
output_dir = r"D:\Download\archive\BSDS100\BSD100_LQ"

os.makedirs(output_dir, exist_ok=True)

def degrade_and_downscale(img):
    # Làm mờ nhẹ (Gaussian blur)
    img_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=0.5)

    # Thêm noise nhẹ
    noise = np.random.normal(0, 3, img.shape).astype(np.uint8)
    img_noisy = cv2.add(img_blur, noise)

    # Nén JPEG (nén nhẹ để giả lập mất chi tiết)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # chất lượng 90 = nhẹ
    _, encimg = cv2.imencode('.jpg', img_noisy, encode_param)
    img_jpeg = cv2.imdecode(encimg, 1)

    # Giảm độ phân giải còn 1/4 (mỗi chiều chia 2)
    h, w = img.shape[:2]
    img_downscaled = cv2.resize(img_jpeg, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)

    return img_downscaled

def process_all():
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Lỗi đọc ảnh: {fname}")
            continue

        img_lqa = degrade_and_downscale(img)

        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, img_lqa)
        print(f"✓ Đã xử lý: {fname} → {img_lqa.shape[1]}x{img_lqa.shape[0]}")

process_all()
