import torch


class ExponentialMovingAverage:
    def __init__(self, parameters, decay=0.999):
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in parameters]

    def update(self, parameters):
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow_params

    def load_state_dict(self, state_dict):
        self.shadow_params = state_dict