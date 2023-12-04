from abc import ABC

import torch.nn as nn

class L1Loss(nn.Module, ABC):
    def __init__(self, gamma=0.8):
        super(L1Loss, self).__init__()
        self.gamma = gamma

    def forward(self, preds, target):

        n_predictions = len(preds)
        valid_mask = (target > 0).detach()
        loss = 0.0

        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)

            diff = target - preds[i]
            diff = diff[valid_mask]
            i_loss = diff.abs().mean()

            loss = loss + i_weight * i_loss

        return loss
