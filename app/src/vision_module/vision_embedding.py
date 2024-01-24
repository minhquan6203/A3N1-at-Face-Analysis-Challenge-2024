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

class Vision_Embedding(nn.Module):
    def __init__(self, config: Dict):
        super(Vision_Embedding,self).__init__()
        self.processor = AutoFeatureExtractor.from_pretrained(config["vision_embedding"]["image_encoder"])
        self.backbone = AutoModel.from_pretrained(config["vision_embedding"]["image_encoder"])
        # freeze all parameters of pretrained model
        if config["vision_embedding"]["freeze"]:
            for param in self.backbone.parameters():
                param.requires_grad = False
        # if config["vision_embedding"]["freeze"]:
        #     freeze_layers = 5
        #     for i, param in enumerate(self.backbone.parameters()):
        #         if i < freeze_layers:
        #             param.requires_grad = False


        self.proj = nn.Linear(config["vision_embedding"]['d_features'], config["vision_embedding"]['d_model'])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config["vision_embedding"]['dropout'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, image_names: List[str]):
        processed_images=self.processor(images=[self.load_image(image_name) for image_name in image_names],return_tensors="pt").to(self.device)
        features = (self.backbone(processed_images.pixel_values).last_hidden_state)
        padding_mask = generate_padding_mask(features, padding_idx=0)
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        return out, padding_mask

    def load_image(self, image_name):
        image = Image.open(image_name).convert('RGB')
        return image