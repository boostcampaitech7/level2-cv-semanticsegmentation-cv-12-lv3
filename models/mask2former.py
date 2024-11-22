import torch
import torch.nn as nn
import numpy as np
from transformers import Mask2FormerForUniversalSegmentation

class Mask2Former(nn.Module):
    def __init__(self, classes=29, encoder = "swin-base"):
        super(Mask2Former, self).__init__()
        
        # encoder 선택에 따른 pretrained 모델 경로 매핑
        encoder_map = {
           "swin-base": "facebook/mask2former-swin-base-ade-semantic",
           "swin-large": "facebook/mask2former-swin-large-ade-semantic",
           "swin-tiny": "facebook/mask2former-swin-tiny-ade-semantic"
        }
        
        if encoder not in encoder_map:
            available_encoders = list(encoder_map.keys())
            raise ValueError(f"Encoder {encoder} not found, Available_encoder:{available_encoders}")
        
        
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            encoder_map[encoder],
            num_labels = classes,
            ignore_mismatched_sizes = True
        
            
        )
    def forward(self, image):
        outputs = self.model(pixel_values=image)
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits
        
        masks_queries_logits_expanded = masks_queries_logits.unsqueeze(2)
        class_queries_logits_expanded = class_queries_logits.unsqueeze(3).unsqueeze(4)
        
        outputs = torch.sum(masks_queries_logits_expanded * class_queries_logits_expanded, dim=1)[:, 1:, ...]
        upsampled_logits = nn.functional.interpolate(
            outputs, 
            size=image.shape[-2:],
            mode="bilinear", 
            align_corners=False
        )
        
        return upsampled_logits
    
    def __str__(self):
       """
       Model prints with number of trainable parameters
       """
       model_parameters = filter(lambda p: p.requires_grad, self.parameters())
       params = sum([np.prod(p.size()) for p in model_parameters])
       return super().__str__() + "\nTrainable parameters: {}".format(params)