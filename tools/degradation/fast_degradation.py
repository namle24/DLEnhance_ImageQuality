import os
import cv2
import numpy as np

input_dir = r"D:\Download\archive\urban100\urban100"
output_dir = r"D:\Download\archive\urban100\urban100_LQ"

os.makedirs(output_dir, exist_ok=True)

def degrade_and_downscale(img):
    img_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=0.2)

    # Nén JPEG nhẹ (chất lượng cao)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 98]  # giữ chi tiết khá tốt
    _, encimg = cv2.imencode('.jpg', img_blur, encode_param)
    img_jpeg = cv2.imdecode(encimg, 1)

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

        img_lq = degrade_and_downscale(img)

        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, img_lq)
        print(f"✓ Đã xử lý: {fname} → {img_lq.shape[1]}x{img_lq.shape[0]}")

process_all()
