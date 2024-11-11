import torch
import torch.nn as nn

from torchvision import models

class TorchvisionModel(nn.Module):
    """
    Baseline Model from torchvision
    """
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 pretrained: bool):
        super(TorchvisionModel, self).__init__()
        self.model = models.segmentation.__dict__[model_name](pretrained=pretrained)

        old_head = self.model.classifier
        last_conv = list(old_head.children())[-1]

        self.model.classifier = nn.Sequential(
            *list(old_head.children())[:-1],
            nn.Conv2d(
                in_channels=last_conv.in_channels,
                out_channels=num_classes,
                kernel_size=last_conv.kernel_size,
                stride=last_conv.stride,
                padding=last_conv.stride,
            ),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)['out']
