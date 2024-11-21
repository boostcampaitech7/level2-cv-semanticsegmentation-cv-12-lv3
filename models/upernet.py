import torch.nn as nn
from transformers import UperNetForSemanticSegmentation
import numpy as np

class UperNet(nn.Module):
   def __init__(self, classes=29, encoder_name="convnext-base"):
       super(UperNet, self).__init__()
       
       # encoder 선택에 따른 pretrained 모델 경로 매핑
       encoder_map = {
           "convnext-base": "openmmlab/upernet-convnext-base",
           "convnext-large": "openmmlab/upernet-convnext-large",
           "swin-base": "openmmlab/upernet-swin-base",
           "swin-large": "openmmlab/upernet-swin-large",
           "swin-tiny": "openmmlab/upernet-swin-tiny"
       }
       
       if encoder_name not in encoder_map:
           available_encoders = list(encoder_map.keys())
           raise ValueError(f"Encoder {encoder_name} not found. Available encoders: {available_encoders}")
           
       self.model = UperNetForSemanticSegmentation.from_pretrained(
           encoder_map[encoder_name],
           num_labels=classes, 
           ignore_mismatched_sizes=True
       )

   def forward(self, image):
       outputs = self.model(pixel_values=image)
       return outputs.logits
       
   def __str__(self):
       """
       Model prints with number of trainable parameters
       """
       model_parameters = filter(lambda p: p.requires_grad, self.parameters())
       params = sum([np.prod(p.size()) for p in model_parameters])
       return super().__str__() + "\nTrainable parameters: {}".format(params)