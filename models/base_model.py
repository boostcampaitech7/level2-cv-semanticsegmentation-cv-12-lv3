import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UnetModel(nn.Module):
    """
    Base Model Unet
    """
    def __init__(self,
                 encoer_name: str,
                 encoder_weights="imagenet",
                 in_channels=3,
                 classes=29):
        super(UnetModel, self).__init__()
        self.model = smp.Unet(encoer_name=encoer_name,
                              encoder_weights=encoder_weights,
                              in_channels=in_channels,
                              classes=classes)

    def forward(self, x: torch.Tensor):
        return self.model(x)