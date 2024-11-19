import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class SegFormerModel(nn.Module):
    """
    SegFormer Model
    """
    def __init__(self, 
                 **kwargs):
        super(SegFormerModel, self).__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(**kwargs)

    def forward(self, x: torch.Tensor):
        return self.model(pixel_values=x).logits