import torch.utils.data as data

class BaseDataset(data.Dataset):
    """Base dataset."""
    def __init__(self, opt):
        self.opt = opt

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
