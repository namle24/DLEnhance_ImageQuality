class PairedImageDataset(Dataset):
    def __init__(self, root_path):
        self.root = root_path
        self.gt_path = os.path.join(root_path, "HR_sub")
        self.low_path = os.path.join(root_path, "LR_light_sub")
        self.very_low_path = os.path.join(root_path, "LR_moderate_sub")

        self.gt_files = sorted(os.listdir(self.gt_path))
        self.low_files = sorted(os.listdir(self.low_path))
        self.very_low_files = sorted(os.listdir(self.very_low_path))

    def __getitem__(self, idx):
        gt = cv2.imread(os.path.join(self.gt_path, self.gt_files[idx]))
        low = cv2.imread(os.path.join(self.low_path, self.low_files[idx]))
        very_low = cv2.imread(os.path.join(self.very_low_path, self.very_low_files[idx]))

        # Chuẩn hóa ảnh về [-1, 1]
        gt = (gt / 127.5) - 1.0
        low = (low / 127.5) - 1.0
        very_low = (very_low / 127.5) - 1.0

        return {
            "gt": torch.FloatTensor(gt).permute(2, 0, 1),
            "low": torch.FloatTensor(low).permute(2, 0, 1),
            "very_low": torch.FloatTensor(very_low).permute(2, 0, 1)
        }