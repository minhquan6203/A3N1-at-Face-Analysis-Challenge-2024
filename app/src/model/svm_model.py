import torch
import torch.nn as nn
import torch.nn.functional as F
from ..vision_module import build_image_embedding
import torch
from torch import nn
from typing import List, Dict, Optional,Any
import numpy as np
from torch.nn import functional as F
from ..data_utils import create_ans_space

class LinearSVM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x):
        outputs = torch.matmul(x, self.weights.t()) + self.bias
        return outputs

class RBFSVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma):
        super(RBFSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))
        self.bn = nn.BatchNorm1d(input_size)  # Thêm BatchNorm1d

    def forward(self, x):
        x = self.bn(x)  # Áp dụng BatchNorm trước khi tính toán kernel
        dists = torch.cdist(x, self.weights, p=2)
        kernel_matrix = torch.exp(-self.gamma * dists ** 2)
        outputs = kernel_matrix + self.bias
        return outputs

class PolySVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma, r, degree):
        super(PolySVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.degree = degree
        self.r = r
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x):
        # dists = torch.cdist(x, self.weights, p=2)
        # kernel_matrix = (self.gamma * dists + self.r) ** self.degree
        kernel_matrix = (self.gamma * torch.mm(x, self.weights.t()) + self.r) ** self.degree
        outputs = kernel_matrix + self.bias
        return outputs
    

class SigmoidSVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma, r):
        super(SigmoidSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.r = r
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))
        self.bn = nn.BatchNorm1d(input_size)  # Thêm BatchNorm1d

    def forward(self, x):
        x = self.bn(x)  # Áp dụng BatchNorm trước khi tính toán kernel
        kernel_matrix = torch.tanh(self.gamma * torch.mm(x, self.weights.t())+ self.r)
        outputs = kernel_matrix  + self.bias
        return outputs


class CustomSVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma, r, degree):
        super(CustomSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.degree = degree
        self.r = r
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x):
        dists = torch.cdist(x, self.weights, p=2)
        kernel_matrix = (self.gamma * dists + self.r) ** self.degree 
        outputs = kernel_matrix + self.bias
        return outputs

def get_kernel(kernel_type,input_size ,num_classes, gamma,r, degree):
    if kernel_type == 'linear':
        return LinearSVM(input_size, num_classes)
    elif kernel_type == 'rbf':
        return RBFSVM(input_size, num_classes, gamma)
    elif kernel_type == 'poly':
        return PolySVM(input_size, num_classes, gamma, r, degree)
    elif kernel_type == 'sigmoid':
        return PolySVM(input_size, num_classes, gamma, r)
    elif kernel_type == 'custom':
        return CustomSVM(input_size, num_classes, gamma, r, degree)
    else:
        raise ValueError('không hỗ trợ kernel này')



class Multi_Task_Model_SVM(nn.Module):
    def __init__(self,config: Dict):

        super(Multi_Task_Model_SVM, self).__init__()
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.gamma = config['svm']['gamma']
        self.kernel_type=config['svm']['kernel_type']
        self.degree = config['svm']['degree']
        self.r=config['svm']['r']
        self.age_space, self.race_space, self.masked_space, self.skintone_space, self.emotion_space, self.gender_space=create_ans_space(config)
        self.vision_embedding=build_image_embedding(config)

        self.classifier_age = get_kernel(self.kernel_type, self.intermediate_dims, len(self.age_space), self.gamma, self.r, self.degree)
        self.classifier_race = get_kernel(self.kernel_type, self.intermediate_dims, len(self.race_space), self.gamma, self.r, self.degree)
        self.classifier_masked = get_kernel(self.kernel_type, self.intermediate_dims, len(self.masked_space), self.gamma, self.r, self.degree)
        self.classifier_skintone = get_kernel(self.kernel_type, self.intermediate_dims, len(self.skintone_space), self.gamma, self.r, self.degree)
        self.classifier_emotion = get_kernel(self.kernel_type, self.intermediate_dims, len(self.emotion_space), self.gamma, self.r, self.degree)
        self.classifier_gender = get_kernel(self.kernel_type, self.intermediate_dims, len(self.gender_space), self.gamma, self.r, self.degree)

        self.attention_weights = nn.Linear(self.intermediate_dims, 1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, images: List[str], age = None, race = None, masked = None,
                skintone = None, emotion = None, gender = None):
        embedded_vision, vison_mask = self.vision_embedding(images)
        feature_attended = self.attention_weights(torch.tanh(embedded_vision))
        attention_weights = torch.softmax(feature_attended, dim=1)
        feature_attended = torch.sum(attention_weights * embedded_vision, dim=1)

        head_age=F.log_softmax(self.classifier_age(feature_attended),dim=-1)
        head_race=F.log_softmax(self.classifier_race(feature_attended),dim=-1)
        head_masked=F.log_softmax(self.classifier_masked(feature_attended),dim=-1)
        head_skintone=F.log_softmax(self.classifier_skintone(feature_attended),dim=-1)
        head_emotion=F.log_softmax(self.classifier_emotion(feature_attended),dim=-1)
        head_gender=F.log_softmax(self.classifier_gender(feature_attended),dim=-1)

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
        

# class Multi_Task_Model_SVM(nn.Module):
#     def __init__(self,config: Dict):

#         super(Multi_Task_Model_SVM, self).__init__()
#         self.intermediate_dims = config["model"]["intermediate_dims"]
#         self.gamma = config['svm']['gamma']
#         self.kernel_type=config['svm']['kernel_type']
#         self.degree = config['svm']['degree']
#         self.r=config['svm']['r']
#         self.age_space, self.race_space, self.masked_space, self.skintone_space, self.emotion_space, self.gender_space=create_ans_space(config)
#         self.vision_embedding=build_image_embedding(config)

#         self.classifier_age = get_kernel(self.kernel_type, self.intermediate_dims, len(self.age_space), self.gamma, self.r, self.degree)
#         self.classifier_race = get_kernel(self.kernel_type, self.intermediate_dims, len(self.race_space), self.gamma, self.r, self.degree)
#         self.classifier_masked = get_kernel(self.kernel_type, self.intermediate_dims, len(self.masked_space), self.gamma, self.r, self.degree)
#         self.classifier_skintone = get_kernel(self.kernel_type, self.intermediate_dims, len(self.skintone_space), self.gamma, self.r, self.degree)
#         self.classifier_emotion = get_kernel(self.kernel_type, self.intermediate_dims, len(self.emotion_space), self.gamma, self.r, self.degree)
#         self.classifier_gender = get_kernel(self.kernel_type, self.intermediate_dims, len(self.gender_space), self.gamma, self.r, self.degree)

#         self.attention_weights = nn.Linear(self.intermediate_dims, 1)
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, images: List[str], age = None, race = None, masked = None,
#                 skintone = None, emotion = None, gender = None):
#         embedded_vision, vison_mask = self.vision_embedding(images)
#         feature_attended = self.attention_weights(torch.tanh(embedded_vision))
#         attention_weights = torch.softmax(feature_attended, dim=1)
#         feature_attended = torch.sum(attention_weights * embedded_vision, dim=1)

#         head_age=F.log_softmax(self.classifier_age(feature_attended),dim=-1)
#         head_race=F.log_softmax(self.classifier_race(feature_attended),dim=-1)
#         head_masked=F.log_softmax(self.classifier_masked(feature_attended),dim=-1)
#         head_skintone=F.log_softmax(self.classifier_skintone(feature_attended),dim=-1)
#         head_emotion=F.log_softmax(self.classifier_emotion(feature_attended),dim=-1)
#         head_gender=F.log_softmax(self.classifier_gender(feature_attended),dim=-1)

#         logits={'head_age': head_age, 'head_race': head_race,
#                 'head_masked': head_masked, 'head_skintone':head_skintone,
#                 'head_emotion': head_emotion, 'head_gender': head_gender}

#         if age is not None and race is not None and masked is not None and skintone is not None and emotion is not None and gender is not None:
#             loss_age = F.binary_cross_entropy_with_logits(head_age, F.one_hot(age, num_classes=len(self.age_space)).float())
#             loss_race = F.binary_cross_entropy_with_logits(head_race, F.one_hot(race, num_classes=len(self.race_space)).float())
#             loss_masked = F.binary_cross_entropy_with_logits(head_masked, F.one_hot(masked, num_classes=len(self.masked_space)).float())
#             loss_skintone = F.binary_cross_entropy_with_logits(head_skintone, F.one_hot(skintone, num_classes=len(self.skintone_space)).float())
#             loss_emotion = F.binary_cross_entropy_with_logits(head_emotion, F.one_hot(emotion, num_classes=len(self.emotion_space)).float())
#             loss_gender = F.binary_cross_entropy_with_logits(head_gender, F.one_hot(gender, num_classes=len(self.gender_space)).float())
#             loss_total = loss_age + loss_race + loss_masked + loss_skintone + loss_emotion + loss_gender

#             loss = {
#                 'loss_age': loss_age,
#                 'loss_race': loss_race,
#                 'loss_masked': loss_masked,
#                 'loss_skintone': loss_skintone,
#                 'loss_emotion': loss_emotion,
#                 'loss_gender': loss_gender,
#                 'loss_total': loss_total
#             }
#             return logits, loss
#         else:
#             return logits