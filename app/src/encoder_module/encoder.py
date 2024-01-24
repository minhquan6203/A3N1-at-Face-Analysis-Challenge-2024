from .attention import MultiHeadAtt
from .feed_foward import PositionWiseFeedForward
import torch
from torch import nn
from typing import List, Dict, Optional,Any
import numpy as np
from torch.nn import functional as F
import math

class SinusoidPositionalEmbedding(nn.Module):

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super(SinusoidPositionalEmbedding, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
        not_mask = (mask == False)
        embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            embed = embed / (embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / self.num_pos_feats)

        pos = embed[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=-1).flatten(-2)

        return pos

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.mhatt = MultiHeadAtt(config)
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, queries, keys, values, attention_mask, **kwargs):
        att = self.mhatt(queries=queries, keys=keys, values=values, attention_mask=attention_mask)
        ff = self.pwff(att)

        return ff

class UniModalEncoder(nn.Module):

    def __init__(self, config):
        super(UniModalEncoder, self).__init__()

        self.pos_embedding = SinusoidPositionalEmbedding(config["encoder"]['d_model'])
        self.layer_norm = nn.LayerNorm(config["encoder"]['d_model'])
        self.d_model = config["encoder"]['d_model']

        self.attn_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config["encoder"]['layers'])])

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor):
        features = self.layer_norm(features) + self.pos_embedding(features)

        for layers in self.attn_layers:
            features = layers(
                queries=features,
                keys=features,
                values=features,
                attention_mask=padding_mask
            )

        return features