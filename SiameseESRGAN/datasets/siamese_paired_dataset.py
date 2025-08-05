import os
from basicsr.data.paired_image_dataset import PairedImageDataset as BaseDataset

class SiamesePairedDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.dataroot_very_low = opt['dataroot_very_low']
        self.very_low_paths = self._scan_paired_folder(opt['dataroot_very_low'])

    def __getitem__(self, index):
        data = super().__getitem__(index)
        very_low_path = self.very_low_paths[index]
        data['very_low'] = self._read_img(very_low_path)
        return data