from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Modified PairedImageDataset to support Siamese training with LQ A and LQ B images."""

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        self.scale = opt.get('scale', 4)
        self.phase = opt.get('phase', 'train')

        # Support siamese inputs
        self.gt_folder = opt['dataroot_gt']
        self.lq_a_folder = opt['dataroot_lq_a']
        self.lq_b_folder = opt['dataroot_lq_b']

        self.paths_gt = sorted(paired_paths_from_folder([self.gt_folder], ['gt'], '{}'))
        self.paths_lq_a = sorted(paired_paths_from_folder([self.lq_a_folder], ['lq_a'], '{}'))
        self.paths_lq_b = sorted(paired_paths_from_folder([self.lq_b_folder], ['lq_b'], '{}'))

        assert len(self.paths_gt) == len(self.paths_lq_a) == len(self.paths_lq_b), 'Dataset size mismatch!'

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load image paths
        gt_path = self.paths_gt[index]['gt_path']
        lq_a_path = self.paths_lq_a[index]['lq_a_path']
        lq_b_path = self.paths_lq_b[index]['lq_b_path']

        # Read images
        img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)
        img_lq_a = imfrombytes(self.file_client.get(lq_a_path, 'lq_a'), float32=True)
        img_lq_b = imfrombytes(self.file_client.get(lq_b_path, 'lq_b'), float32=True)

        # Random crop
        if self.phase == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq_a = paired_random_crop(img_gt, img_lq_a, gt_size, self.scale, gt_path)
            _, img_lq_b = paired_random_crop(img_gt, img_lq_b, gt_size, self.scale, gt_path)

            # Augment
            img_gt, img_lq_a, img_lq_b = augment([img_gt, img_lq_a, img_lq_b],
                                                 self.opt.get('use_hflip', True),
                                                 self.opt.get('use_rot', True))

        # BGR to RGB, HWC to CHW, to tensor
        img_gt, img_lq_a, img_lq_b = img2tensor([img_gt, img_lq_a, img_lq_b], bgr2rgb=True, float32=True)

        # Normalize if needed
        if self.mean is not None or self.std is not None:
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lq_a, self.mean, self.std, inplace=True)
            normalize(img_lq_b, self.mean, self.std, inplace=True)

        return {
            'gt': img_gt,
            'lq_a': img_lq_a,
            'lq_b': img_lq_b,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths_gt)
