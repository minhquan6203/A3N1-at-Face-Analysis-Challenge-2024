import torch
from torch import nn
import os
from PIL import Image
from transformers import AutoModel, AutoFeatureExtractor, AutoProcessor
from typing import List, Dict, Optional,Any


def generate_padding_mask(sequences: torch.Tensor, padding_idx: int) -> torch.BoolTensor:
    if sequences is None:
        return None

    if len(sequences.shape) == 2: # (bs, seq_len)
        __seq = sequences.unsqueeze(dim=-1) # (bs, seq_len, 1)
    else:
        __seq = sequences

    mask = (torch.sum(__seq, dim=-1) == (padding_idx*__seq.shape[-1])).long() * -10e4 # (b_s, seq_len)
    return mask.unsqueeze(1).unsqueeze(1) # (bs, 1, 1, seq_len)

from torchvision import models, transforms

class ResNet_Embedding(nn.Module):
    def __init__(self, config: Dict):
        super(ResNet_Embedding, self).__init__()
        resnet_model = models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet_model.children())[:-2])

        self.proj = nn.Linear(config["vision_embedding"]['d_features'], config["vision_embedding"]['d_model'])
        if config["vision_embedding"]["freeze"]:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config["vision_embedding"]['dropout'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, image_names: List[str]):
        processed_images = [self.load_and_preprocess_image(image_name) for image_name in image_names]
        processed_images = torch.stack(processed_images).to(self.device)

        features = self.backbone(processed_images)
        features = features.view(features.size(0),features.size(1), -1).permute(0, 2, 1)
        padding_mask = generate_padding_mask(features, padding_idx=0)
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        return out, padding_mask

    def load_and_preprocess_image(self, image_name):
        image = Image.open(image_name).convert('RGB')
        image = self.image_transform(image)
        return image
