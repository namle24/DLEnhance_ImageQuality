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
        self.use_flip = opt.get('use_flip', True)
        self.use_rot = opt.get('use_rot', True)
        self.phase = opt.get('phase', 'train')

        self.paths_gt = []
        self.paths_lq_a = []
        self.paths_lq_b = []

        meta_file = opt['meta_info_file']
        with open(meta_file, 'r') as f:
            for line in f:
                gt, lq_a, lq_b = line.strip().split(',')
                self.paths_gt.append(os.path.join(opt['dataroot_gt'], gt.strip()))
                self.paths_lq_a.append(os.path.join(opt['dataroot_lq_a'], lq_a.strip()))
                self.paths_lq_b.append(os.path.join(opt['dataroot_lq_b'], lq_b.strip()))

        self.file_client = None
        print(f'[INFO] PairedSiameseImageDataset: {len(self.paths_gt)} samples loaded.')

    def __getitem__(self, index):
        try:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

            # Lấy đường dẫn ảnh
            gt_path = self.paths_gt[index]
            lq_a_path = self.paths_lq_a[index]
            lq_b_path = self.paths_lq_b[index]

            # Load ảnh dạng bytes
            gt_bytes = self.file_client.get(gt_path, 'gt')
            lq_a_bytes = self.file_client.get(lq_a_path, 'lq_a')
            lq_b_bytes = self.file_client.get(lq_b_path, 'lq_b')

            img_gt = imfrombytes(gt_bytes, float32=True)
            img_lq_a = imfrombytes(lq_a_bytes, float32=True)
            img_lq_b = imfrombytes(lq_b_bytes, float32=True)

            # Check nếu bất kỳ ảnh nào rỗng
            if img_gt is None or img_lq_a is None or img_lq_b is None:
                raise ValueError("One of the images is None.")

            # Crop và augment nếu training
            if self.phase == 'train' and self.gt_size is not None:
                h, w = img_gt.shape[:2]
                if h < self.gt_size or w < self.gt_size:
                    raise ValueError(f"Image too small: {gt_path} ({h}x{w})")
                rnd_h = random.randint(0, h - self.gt_size)
                rnd_w = random.randint(0, w - self.gt_size)
                img_gt = img_gt[rnd_h:rnd_h + self.gt_size, rnd_w:rnd_w + self.gt_size, :]
                img_lq_a = img_lq_a[rnd_h:rnd_h + self.gt_size, rnd_w:rnd_w + self.gt_size, :]
                img_lq_b = img_lq_b[rnd_h:rnd_h + self.gt_size, rnd_w:rnd_w + self.gt_size, :]
                img_gt, img_lq_a, img_lq_b = augment([img_gt, img_lq_a, img_lq_b], self.use_flip, self.use_rot)

            # Convert sang tensor
            img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
            img_lq_a = img2tensor(img_lq_a, bgr2rgb=True, float32=True)
            img_lq_b = img2tensor(img_lq_b, bgr2rgb=True, float32=True)

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
            print(f"[WARNING] Error loading index {index}: {e}")
            # Nếu lỗi thì đệ quy lấy ảnh kế tiếp (vòng lại nếu cuối)
            return self.__getitem__((index + 1) % self.__len__())


    def __len__(self):
        return len(self.paths_gt)
