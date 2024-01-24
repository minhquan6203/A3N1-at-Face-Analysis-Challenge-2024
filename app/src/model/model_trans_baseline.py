from ..encoder_module import UniModalEncoder
from ..vision_module import build_image_embedding
import torch
from torch import nn
from typing import List, Dict, Optional,Any
import numpy as np
from torch.nn import functional as F
from ..data_utils import create_ans_space

class Multi_Task_Model_CE(nn.Module):
    def __init__(self,config: Dict):

        super(Multi_Task_Model_CE, self).__init__()
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.encoder = UniModalEncoder(config)
        self.age_space, self.race_space, self.masked_space, self.skintone_space, self.emotion_space, self.gender_space=create_ans_space(config)
        self.vision_embedding=build_image_embedding(config)

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
        encoded_feature = self.encoder(embedded_vision, vison_mask)
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
        

#         if age is not None and race is not None and masked is not None and skintone is not None and emotion is not None and gender is not None:
#             loss_age = F.binary_cross_entropy_with_logits(head_age, F.one_hot(age, num_classes=len(self.age_space)).float())
#             loss_race = F.binary_cross_entropy_with_logits(head_race, F.one_hot(race, num_classes=len(self.race_space)).float())
#             loss_masked = F.binary_cross_entropy_with_logits(head_masked, F.one_hot(masked, num_classes=len(self.masked_space)).float())
#             loss_skintone = F.binary_cross_entropy_with_logits(head_skintone, F.one_hot(skintone, num_classes=len(self.skintone_space)).float())
#             loss_emotion = F.binary_cross_entropy_with_logits(head_emotion, F.one_hot(emotion, num_classes=len(self.emotion_space)).float())
#             loss_gender = F.binary_cross_entropy_with_logits(head_gender, F.one_hot(gender, num_classes=len(self.gender_space)).float())
#             loss_total = loss_age + loss_race + loss_masked + loss_skintone + loss_emotion + loss_gender