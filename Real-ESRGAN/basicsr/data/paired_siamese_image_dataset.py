import os
import random
import cv2
import numpy as np

from basicsr.data.base_dataset import BaseDataset
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import augment

@DATASET_REGISTRY.register()
class PairedSiameseImageDataset(BaseDataset):
    """Dataset for Siamese Real-ESRGAN using triplet (GT, LQ_A, LQ_B) from meta_info_file."""

    def __init__(self, opt):
        super().__init__(opt)
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', [0.0, 0.0, 0.0])
        self.std = opt.get('std', [1.0, 1.0, 1.0])
        self.gt_size = opt.get('gt_size', None)
        self.scale = opt.get('scale', 4)
        if isinstance(self.gt_size, str):
            if self.gt_size.lower() == 'none':
                self.gt_size = None
            else:
                self.gt_size = int(self.gt_size)
        self.use_flip = opt.get('use_hflip', True)
        self.use_rot = opt.get('use_rot', True)
        self.phase = opt.get('phase', 'train')

        self.paths_gt = []
        self.paths_lq_a = []
        self.paths_lq_b = []

        meta_file = opt['meta_info_file']
        with open(meta_file, 'r') as f:
            for line in f:
                gt, lq_a, lq_b = line.strip().split(',')
                gt_path = os.path.join(opt['dataroot_gt'], gt.strip())
                lq_a_path = os.path.join(opt['dataroot_lq_a'], lq_a.strip())
                lq_b_path = os.path.join(opt['dataroot_lq_b'], lq_b.strip())

                if not os.path.isfile(gt_path):
                    print(f"[ERROR] GT file not found: {gt_path}")
                    continue
                if not os.path.isfile(lq_a_path):
                    print(f"[ERROR] LQ_A file not found: {lq_a_path}")
                    continue
                if not os.path.isfile(lq_b_path):
                    print(f"[ERROR] LQ_B file not found: {lq_b_path}")
                    continue

                self.paths_gt.append(gt_path)
                self.paths_lq_a.append(lq_a_path)
                self.paths_lq_b.append(lq_b_path)

        if not self.paths_gt:
            raise ValueError("[ERROR] No valid image triplets found in meta_info_file!")

        self.file_client = None
        print(f'[INFO] PairedSiameseImageDataset: {len(self.paths_gt)} samples loaded.')

    def __getitem__(self, index):
        try:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

            gt_path = self.paths_gt[index]
            lq_a_path = self.paths_lq_a[index]
            lq_b_path = self.paths_lq_b[index]

            # Load images
            img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)
            img_lq_a = imfrombytes(self.file_client.get(lq_a_path, 'lq_a'), float32=True)
            img_lq_b = imfrombytes(self.file_client.get(lq_b_path, 'lq_b'), float32=True)

            # Check image validity
            for name, img, path in [('GT', img_gt, gt_path), ('LQ_A', img_lq_a, lq_a_path), ('LQ_B', img_lq_b, lq_b_path)]:
                if img is None or img.size == 0:
                    raise ValueError(f"Invalid image {name}: {path}")
                if len(img.shape) != 3 or img.shape[2] != 3:
                    raise ValueError(f"Image {name} is not RGB: {path}")

            # Calculate expected LQ size
            h_gt, w_gt = img_gt.shape[:2]
            expected_h_lq = h_gt // self.scale
            expected_w_lq = w_gt // self.scale

            # Resize LQ images to expected size if needed (using INTER_AREA for downscaling)
            if img_lq_a.shape[:2] != (expected_h_lq, expected_w_lq):
                img_lq_a = cv2.resize(img_lq_a, (expected_w_lq, expected_h_lq), interpolation=cv2.INTER_AREA)
            if img_lq_b.shape[:2] != (expected_h_lq, expected_w_lq):
                img_lq_b = cv2.resize(img_lq_b, (expected_w_lq, expected_h_lq), interpolation=cv2.INTER_AREA)

            # For training, crop patches
            if self.phase == 'train' and self.gt_size is not None:
                # Calculate LQ crop size
                lq_crop_size = self.gt_size // self.scale
                
                # Ensure LQ images are large enough
                if img_lq_a.shape[0] < lq_crop_size or img_lq_a.shape[1] < lq_crop_size:
                    # If image is too small, first pad then crop
                    pad_h = max(0, lq_crop_size - img_lq_a.shape[0])
                    pad_w = max(0, lq_crop_size - img_lq_a.shape[1])
                    
                    img_lq_a = cv2.copyMakeBorder(img_lq_a, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                    img_lq_b = cv2.copyMakeBorder(img_lq_b, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                    img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h * self.scale, 0, pad_w * self.scale, cv2.BORDER_REFLECT)

                # Random crop coordinates for LQ
                top = random.randint(0, img_lq_a.shape[0] - lq_crop_size)
                left = random.randint(0, img_lq_a.shape[1] - lq_crop_size)
                
                # Crop LQ images
                img_lq_a = img_lq_a[top:top + lq_crop_size, left:left + lq_crop_size, :]
                img_lq_b = img_lq_b[top:top + lq_crop_size, left:left + lq_crop_size, :]
                
                # Calculate corresponding GT crop coordinates and crop
                top_gt, left_gt = top * self.scale, left * self.scale
                img_gt = img_gt[top_gt:top_gt + self.gt_size, left_gt:left_gt + self.gt_size, :]

            # Data augmentation (sync for all images)
            img_gt, img_lq_a, img_lq_b = augment([img_gt, img_lq_a, img_lq_b], self.use_flip, self.use_rot)

            # Convert to tensor
            img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
            img_lq_a = img2tensor(img_lq_a, bgr2rgb=True, float32=True)
            img_lq_b = img2tensor(img_lq_b, bgr2rgb=True, float32=True)

            # Normalize
            for t in [img_gt, img_lq_a, img_lq_b]:
                for c in range(3):
                    t[c, :, :] = (t[c, :, :] - self.mean[c]) / self.std[c]

            return {
                'gt': img_gt,
                'lq_a': img_lq_a,
                'lq_b': img_lq_b,
                'gt_path': gt_path,
                'lq_a_path': lq_a_path,
                'lq_b_path': lq_b_path
            }

        except Exception as e:
            print(f"Error loading index {index}: {e}")
            return self.__getitem__((index + 1) % self.__len__())

    def __len__(self):
        return len(self.paths_gt)