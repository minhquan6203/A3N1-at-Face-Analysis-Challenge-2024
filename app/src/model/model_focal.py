from ..encoder_module import UniModalEncoder
from ..vision_module import build_image_embedding
import torch
from torch import nn
from typing import List, Dict, Optional,Any
import numpy as np
from torch.nn import functional as F
from typing import Optional, Sequence
from torch import Tensor
from ..data_utils import create_ans_space
from torch.autograd import Variable
import pandas as pd

def calculate_class_weights(column_name: str, config: Dict):
    train_path=config['data']['train_dataset']
    train_df=pd.read_csv(train_path)
    class_counts = train_df[column_name].value_counts().sort_index()
    total_samples = len(train_df)
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    class_weights /= class_weights.sum()
    class_weights_tensor = torch.Tensor(class_weights.values)
    return class_weights_tensor

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.

    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.

    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl

class Multi_Task_Model_Focal(nn.Module):
    def __init__(self,config: Dict):

        super(Multi_Task_Model_Focal, self).__init__()
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.encoder = UniModalEncoder(config)
        self.age_space, self.race_space, self.masked_space, self.skintone_space, self.emotion_space, self.gender_space=create_ans_space(config)
        self.image_embedding=build_image_embedding(config)

        self.classifier_age = nn.Linear(self.intermediate_dims, len(self.age_space))
        self.classifier_race = nn.Linear(self.intermediate_dims, len(self.race_space))
        self.classifier_masked = nn.Linear(self.intermediate_dims, len(self.masked_space))
        self.classifier_skintone = nn.Linear(self.intermediate_dims, len(self.skintone_space))
        self.classifier_emotion = nn.Linear(self.intermediate_dims, len(self.emotion_space))
        self.classifier_gender = nn.Linear(self.intermediate_dims, len(self.gender_space))

        self.criterion = nn.CrossEntropyLoss()
        
        self.race_weight=torch.tensor([0.15, 0.15, 0.7])
        self.masked_weight=torch.tensor([0.85,0.15])
        self.gender_weight=torch.tensor([0.4, 0.6])

        self.age_weight=torch.tensor([0.1, 0.1, 0.3, 0.15, 0.15, 0.2])
        self.skintone_weight=torch.tensor([0.4, 0.15, 0.3, 0.15])
        self.emotion_weight=torch.tensor([0.15, 0.2, 0.2, 0.1, 0.1, 0.1, 0.15])
        # self.age_weight.fill_(0.25)
        # self.skintone_weight.fill_(0.25)
        # self.emotion_weight.fill_(0.25)
        
        self.focal_loss_age = FocalLoss(alpha=self.age_weight,gamma=2)
        self.focal_loss_skintone = FocalLoss(alpha=self.skintone_weight,gamma=2)
        self.focal_loss_emotion = FocalLoss(alpha=self.emotion_weight,gamma=2)

        self.focal_loss_race = FocalLoss(alpha=self.race_weight,gamma=2)
        self.focal_loss_masked = FocalLoss(alpha=self.masked_weight,gamma=2)
        self.focal_loss_gender = FocalLoss(alpha=self.gender_weight,gamma=2)


    def forward(self, images: List[str], age = None, race = None, masked = None,
                skintone = None, emotion = None, gender = None):
        embedded_vision, vison_mask = self.image_embedding(images)
        encoded_feature = self.encoder(embedded_vision, vison_mask)
        head_age=torch.mean(self.classifier_age(encoded_feature),dim=1)
        head_race=torch.mean(self.classifier_race(encoded_feature),dim=1)
        head_masked=torch.mean(self.classifier_masked(encoded_feature),dim=1)
        head_skintone=torch.mean(self.classifier_skintone(encoded_feature),dim=1)
        head_emotion=torch.mean(self.classifier_emotion(encoded_feature),dim=1)
        head_gender=torch.mean(self.classifier_gender(encoded_feature),dim=1)

        logits={'head_age': head_age, 'head_race': head_race,
                'head_masked': head_masked, 'head_skintone':head_skintone,
                'head_emotion': head_emotion, 'head_gender': head_gender}

        if age is not None and race is not None and masked is not None and skintone is not None and emotion is not None and gender is not None:
            loss_age = self.focal_loss_age(head_age, age)
            loss_race = self.focal_loss_race(head_race, race)
            loss_masked = self.focal_loss_masked(head_masked, masked)
            loss_skintone = self.focal_loss_skintone(head_skintone, skintone)
            loss_emotion = self.focal_loss_emotion(head_emotion, emotion)
            loss_gender = self.focal_loss_gender(head_gender, gender)
            loss_total = loss_age + loss_race + loss_masked + loss_skintone + loss_emotion + loss_gender

            loss={'loss_age': loss_age, 'loss_race': loss_race,
                  'loss_masked': loss_masked, 'loss_skintone':loss_skintone,
                  'loss_emotion': loss_emotion, 'loss_gender': loss_gender,
                  'loss_total': loss_total}
            return logits, loss
        else:
            return logits