import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, losses, weights=None):
        super(CombinedLoss, self).__init__()
        self.losses = losses
        self.weights = weights or [1.0] * len(losses)

    def forward(self, predictions, targets):
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(predictions, targets)
        return total_loss