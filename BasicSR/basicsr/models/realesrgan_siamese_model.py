import torch
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class RealESRGANSiameseModel(SRGANModel):
    def __init__(self, opt):
        super(RealESRGANSiameseModel, self).__init__(opt)
    
    def get_intermediate_features(self, lq):
        """Extract intermediate features from the generator network."""
        x = lq
        features = []
        for block_idx, block in enumerate(self.net_g.body):
            x = block(x)
            if block_idx in [5, 11, 17]:
                features.append(x)
        return features