from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired Siamese dataset for image restoration.

    Reads LQ_A, LQ_B, and GT image triplets.

    Required keys in opt:
        - dataroot_gt
        - dataroot_lq_a
        - dataroot_lq_b
        - io_backend
        - phase: train/val
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt['dataroot_gt']
        self.lq_a_folder = opt['dataroot_lq_a']
        self.lq_b_folder = opt['dataroot_lq_b']

        gt_paths = sorted(os.listdir(self.gt_folder))
        lq_a_paths = sorted(os.listdir(self.lq_a_folder))
        lq_b_paths = sorted(os.listdir(self.lq_b_folder))

        assert len(gt_paths) == len(lq_a_paths) == len(lq_b_paths), 'Số lượng ảnh không khớp!'

        # tạo danh sách dict kiểu giống code gốc
        self.paths = []
        for gt, lqa, lqb in zip(gt_paths, lq_a_paths, lq_b_paths):
            self.paths.append({
                'gt_path': os.path.join(self.gt_folder, gt),
                'lq_a_path': os.path.join(self.lq_a_folder, lqa),
                'lq_b_path': os.path.join(self.lq_b_folder, lqb)
            })

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt.get('gt_size', 256)

        paths = self.paths[index]
        gt_path, lq_a_path, lq_b_path = paths['gt_path'], paths['lq_a_path'], paths['lq_b_path']

        img_gt = imfrombytes(self.file_client.get(gt_path), float32=True)
        img_lq_a = imfrombytes(self.file_client.get(lq_a_path), float32=True)
        img_lq_b = imfrombytes(self.file_client.get(lq_b_path), float32=True)

        if self.opt['phase'] == 'train':
            img_gt, img_lq_a = paired_random_crop(img_gt, img_lq_a, gt_size, scale, gt_path)
            _, img_lq_b = paired_random_crop(img_gt, img_lq_b, gt_size, scale, gt_path)
            img_gt, img_lq_a = augment([img_gt, img_lq_a], self.opt.get('use_hflip', True), self.opt.get('use_rot', True))
            _, img_lq_b = augment([img_gt, img_lq_b], self.opt.get('use_hflip', True), self.opt.get('use_rot', True))

        # to tensor
        img_gt, img_lq_a, img_lq_b = img2tensor([img_gt, img_lq_a, img_lq_b], bgr2rgb=True, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lq_a, self.mean, self.std, inplace=True)
            normalize(img_lq_b, self.mean, self.std, inplace=True)

        return {
            'gt': img_gt,
            'lq_a': img_lq_a,
            'lq_b': img_lq_b,
            'gt_path': gt_path,
            'lq_a_path': lq_a_path,
            'lq_b_path': lq_b_path
        }

    def __len__(self):
        return len(self.paths)
