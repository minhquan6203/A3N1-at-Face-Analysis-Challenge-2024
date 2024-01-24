from ..encoder_module import UniModalEncoder
import torch
from torch import nn
from typing import List, Dict, Optional,Any
import numpy as np
from torch.nn import functional as F
from ..data_utils import create_ans_space
import os
from PIL import Image
from transformers import AutoModel, AutoFeatureExtractor, AutoProcessor


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
        resnet_model = models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet_model.children())[:-2])

        self.proj = nn.Linear(config["res_embedding"]['d_features'], config["res_embedding"]['d_model'])
        if config["res_embedding"]["freeze"]:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config["res_embedding"]['dropout'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_folder = config["data"]["images_folder"]

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
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        return image
    

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
        self.image_folder = config["data"]["images_folder"]

    def forward(self, image_names: List[str]):
        processed_images=self.processor(images=[self.load_image(image_name) for image_name in image_names],return_tensors="pt").to(self.device)
        features = (self.backbone(processed_images.pixel_values).last_hidden_state)
        padding_mask = generate_padding_mask(features, padding_idx=0)
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        return out, padding_mask

    def load_image(self, image_name):
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        return image


class Multi_Task_Model_VIT_RES(nn.Module):
    def __init__(self,config: Dict):

        super(Multi_Task_Model_VIT_RES, self).__init__()
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.encoder = UniModalEncoder(config)
        self.age_space, self.race_space, self.masked_space, self.skintone_space, self.emotion_space, self.gender_space=create_ans_space(config)
        self.vision_embedding=Vision_Embedding(config)
        self.resnet_embedding=ResNet_Embedding(config)

        self.classifier_age = nn.Linear(self.intermediate_dims, len(self.age_space))
        self.classifier_race = nn.Linear(self.intermediate_dims, len(self.race_space))
        self.classifier_masked = nn.Linear(self.intermediate_dims, len(self.masked_space))
        self.classifier_skintone = nn.Linear(self.intermediate_dims, len(self.skintone_space))
        self.classifier_emotion = nn.Linear(self.intermediate_dims, len(self.emotion_space))
        self.classifier_gender = nn.Linear(self.intermediate_dims, len(self.gender_space))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, images: List[str], age = None, race = None, masked = None,
                skintone = None, emotion = None, gender = None):
        embedded_vision, vison_mask = self.vision_embedding(images)
        embedded_res, res_mask = self.resnet_embedding(images)
        embeded = torch.cat([embedded_vision,embedded_res],dim=1)
        mask=torch.cat([vison_mask.squeeze(1).squeeze(1),res_mask.squeeze(1).squeeze(1)],dim=1)
        mask=mask.unsqueeze(1).unsqueeze(2)
        encoded_feature = self.encoder(embeded, mask)
        head_age=F.log_softmax(torch.mean(self.classifier_age(encoded_feature),dim=1),dim=-1)
        head_race=F.log_softmax(torch.mean(self.classifier_race(encoded_feature),dim=1),dim=-1)
        head_masked=F.log_softmax(torch.mean(self.classifier_masked(encoded_feature),dim=1),dim=-1)
        head_skintone=F.log_softmax(torch.mean(self.classifier_skintone(encoded_feature),dim=1),dim=-1)
        head_emotion=F.log_softmax(torch.mean(self.classifier_emotion(encoded_feature),dim=1),dim=-1)
        head_gender=F.log_softmax(torch.mean(self.classifier_gender(encoded_feature),dim=1),dim=-1)

        logits={'head_age': head_age, 'head_race': head_race,
                'head_masked': head_masked, 'head_skintone':head_skintone,
                'head_emotion': head_emotion, 'head_gender': head_gender}

        if age is not None and race is not None and masked is not None and skintone is not None and emotion is not None and gender is not None:
            loss_age = self.criterion(head_age, age)
            loss_race = self.criterion(head_race, race)
            loss_masked = self.criterion(head_masked, masked)
            loss_skintone = self.criterion(head_skintone, skintone)
            loss_emotion = self.criterion(head_emotion, emotion)
            loss_gender = self.criterion(head_gender, gender)
            loss_total = loss_age + loss_race + loss_masked + loss_skintone + loss_emotion + loss_gender

            loss={'loss_age': loss_age, 'loss_race': loss_race,
                  'loss_masked': loss_masked, 'loss_skintone':loss_skintone,
                  'loss_emotion': loss_emotion, 'loss_gender': loss_gender,
                  'loss_total': loss_total}
            return logits, loss
        else:
            return logits