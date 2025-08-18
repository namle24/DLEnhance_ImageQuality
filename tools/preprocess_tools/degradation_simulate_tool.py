import os
import cv2
import glob
import argparse
import numpy as np
from pathlib import Path

def random_rescale(img):
    h, w = img.shape[:2]
    scale = np.random.uniform(0.9, 1.0)  # giảm ít hơn (nhẹ hơn)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    return img

def random_blur(img):
    if np.random.rand() < 0.7:  # ưu tiên Gaussian blur nhẹ
        ksize = np.random.choice([1, 3])
        img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0)
    else:
        ksize = 5
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize - 1)/2), :] = np.ones(ksize)
        kernel /= ksize
        img = cv2.filter2D(img, -1, kernel)
    return img

def random_noise(img):
    if np.random.rand() < 0.7:
        noise = np.random.normal(0, 4, img.shape).astype(np.float32)  # σ nhỏ hơn
        img = cv2.add(img.astype(np.float32), noise)
    else:
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img * vals) / float(vals)
        img = np.clip(noisy, 0, 255)
    return img.astype(np.uint8)

def random_jpeg(img):
    """Thêm JPEG artifacts nhẹ hơn."""
    quality = np.random.randint(90, 95)  # ít suy giảm hơn
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    img = cv2.imdecode(encimg, 1)
    return img

def degrade(img):
    img = random_rescale(img)
    img = random_blur(img)
    img = random_noise(img)
    img = random_jpeg(img)
    return img

def main(input_dir, output_dir, scale_factor):
    os.makedirs(output_dir, exist_ok=True)
    img_list = glob.glob(os.path.join(input_dir, '*.*'))

    for idx, img_path in enumerate(img_list):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Lỗi đọc ảnh: {img_path}")
            continue

        h, w = img.shape[:2]

        # pipeline suy giảm
        lq = degrade(img)

        # giảm độ phân giải theo scale
        lq = cv2.resize(lq, (w // scale_factor, h // scale_factor), interpolation=cv2.INTER_CUBIC)

        filename = Path(img_path).stem
        save_path = os.path.join(output_dir, f"{filename}.png")
        cv2.imwrite(save_path, lq)

        if idx % 20 == 0:
            print(f"Đã xử lý {idx}/{len(img_list)} ảnh")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Folder chứa ảnh gốc")
    parser.add_argument("--output", type=str, required=True, help="Folder để lưu ảnh LQ")
    parser.add_argument("--scale", type=int, default=4, help="Tỉ lệ giảm độ phân giải (mặc định = 4)")
    args = parser.parse_args()

    main(args.input, args.output, args.scale)
