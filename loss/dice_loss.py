import torch
import torch.nn as nn

class CustomDiceLoss(nn.Module):
    """
    Dice Loss
    """
    def __init__(self, smooth=1e-5):
        super(CustomDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice