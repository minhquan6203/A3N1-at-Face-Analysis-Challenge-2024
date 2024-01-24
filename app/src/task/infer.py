import torch
from torch import nn
from typing import List, Dict, Optional,Any
import numpy as np
from torch.nn import functional as F
from torch import Tensor
from data_utils.load_data import Get_Loader,create_ans_space
from model.build_model import build_model
import os
from tqdm import tqdm
import shutil
import pandas as pd
class Predict:
    def __init__(self, config: Dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path=os.path.join(config["train"]["output_dir"], "best_model.pth")
        self.age_space, self.race_space, self.masked_space, self.skintone_space, self.emotion_space, self.gender_space=create_ans_space(config)
        self.model = build_model(config)
        self.model.to(self.device)
        self.dataloader = Get_Loader(config)

    def predict_submission(self):
        # Load the model
        print("Loading the best model...")
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Obtain the prediction from the model
        print("Obtaining predictions...")
        test =self.dataloader.load_test()
        ids=[]
        pred_age=[]
        pred_race=[]
        pred_masked=[]
        pred_skintone=[]
        pred_emotion=[]
        pred_gender=[]

        self.model.eval()
        with torch.no_grad():
            for it, item in enumerate(tqdm(test)):
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                    logits = self.model(item['image_name'])
                pred_age.extend(logits['head_age'].argmax(-1))
                pred_race.extend(logits['head_race'].argmax(-1))
                pred_masked.extend(logits['head_masked'].argmax(-1))
                pred_skintone.extend(logits['head_skintone'].argmax(-1))
                pred_emotion.extend(logits['head_emotion'].argmax(-1))
                pred_gender.extend(logits['head_gender'].argmax(-1))
                ids.extend(item['image_name'])

        pred_age=[self.age_space[i] for i in pred_age]
        pred_race=[self.race_space[i] for i in pred_race]
        pred_masked=[self.masked_space[i] for i in pred_masked]
        pred_skintone=[self.skintone_space[i] for i in pred_skintone]
        pred_emotion=[self.emotion_space[i] for i in pred_emotion]
        pred_gender=[self.gender_space[i] for i in pred_gender]
        data = {'crop_name':ids,
                'age': pred_age,
                'race': pred_race,
                'masked': pred_masked,
                'skintone': pred_skintone,
                'emotion': pred_emotion,
                'gender': pred_gender}
        df = pd.DataFrame(data)
        df.to_csv('submission.csv', index=False)