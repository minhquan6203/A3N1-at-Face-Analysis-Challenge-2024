from .resnet_embedding import ResNet_Embedding
from .vision_embedding import Vision_Embedding
from typing import List, Dict, Optional,Any

def build_image_embedding(config:Dict):
    if config['vision_embedding']['type']=='VIT':
        return Vision_Embedding(config)
    if config['vision_embedding']['type']=='ResNet':
        return ResNet_Embedding(config)