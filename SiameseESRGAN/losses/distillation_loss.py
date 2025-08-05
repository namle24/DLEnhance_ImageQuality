import torch.nn as nn

class DistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_feat, teacher_feat):
        return self.mse(student_feat, teacher_feat.detach())