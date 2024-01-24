import torch
from torch import nn
from typing import List, Dict, Optional,Any
import numpy as np
from torch.nn import functional as F
from typing import Optional, Sequence
from torch import Tensor
from data_utils.load_data import Get_Loader
from model.build_model import build_model
from eval_metric.evaluate import ScoreCalculator
import torch.optim as optim
import os
from tqdm import tqdm
class Classify_Task:
    def __init__(self, config: Dict):
        self.num_epochs = config['train']['num_train_epochs']
        self.patience = config['train']['patience']
        self.learning_rate = config['train']['learning_rate']
        self.save_path = config['train']['output_dir']
        self.best_metric= config['train']['metric_for_best_model']
        self.weight_decay=config['train']['weight_decay']
        self.dataloader = Get_Loader(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model=build_model(config)
        self.base_model.to(self.device)
        self.compute_score = ScoreCalculator()
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()
        lambda1 = lambda epoch: 0.95 ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

    def training(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        train, valid = self.dataloader.load_train_dev()

        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['self.optimizer_state_dict'])
            print('loaded the last saved model!!!')
            initial_epoch = checkpoint['epoch'] + 1
            print(f"continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("first time training!!!")

        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_score = checkpoint['score']
        else:
            best_score = 0.

        threshold=0
        self.base_model.train()
        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            avg_valid_acc = 0.
            avg_valid_f1 = 0.

            age_acc=0.
            race_acc=0.
            masked_acc=0.
            skintone_acc=0.
            emotion_acc=0.
            gender_acc=0.

            age_f1=0.
            race_f1=0.
            masked_f1=0.
            skintone_f1=0.
            emotion_f1=0.
            gender_f1=0.

            train_loss = 0.
            for it, item in enumerate(tqdm(train)):
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                    age = item['age'].to(dtype=torch.long, device=self.device)
                    race = item['race'].to(dtype=torch.long, device=self.device)
                    masked = item['masked'].to(dtype=torch.long, device=self.device)
                    skintone = item['skintone'].to(dtype=torch.long, device=self.device)
                    emotion = item['emotion'].to(dtype=torch.long, device=self.device)
                    gender = item['gender'].to(dtype=torch.long, device=self.device)
                    logits, loss = self.base_model(item['image_name'], age,race,masked,skintone,emotion,gender)
                self.scaler.scale(loss['loss_total']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                train_loss += loss['loss_total'].item()

            self.scheduler.step()
            train_loss /=len(train)

            with torch.no_grad():
                for it, item in enumerate(tqdm(valid)):
                    with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                        logits = self.base_model(item['image_name'])
                    pred_age = logits['head_age'].argmax(-1)
                    pred_race = logits['head_race'].argmax(-1)
                    pred_masked = logits['head_masked'].argmax(-1)
                    pred_skintone = logits['head_skintone'].argmax(-1)
                    pred_emotion = logits['head_emotion'].argmax(-1)
                    pred_gender = logits['head_gender'].argmax(-1)

                    age_acc+=self.compute_score.acc(item['age'],pred_age)
                    race_acc+=self.compute_score.acc(item['race'],pred_race)
                    masked_acc+=self.compute_score.acc(item['masked'],pred_masked)
                    skintone_acc+=self.compute_score.acc(item['skintone'],pred_skintone)
                    emotion_acc+=self.compute_score.acc(item['emotion'],pred_emotion)
                    gender_acc+=self.compute_score.acc(item['gender'],pred_gender)

                    age_f1+=self.compute_score.f1(item['age'],pred_age)
                    race_f1+=self.compute_score.f1(item['race'],pred_race)
                    masked_f1+=self.compute_score.f1(item['masked'],pred_masked)
                    skintone_f1+=self.compute_score.f1(item['skintone'],pred_skintone)
                    emotion_f1+=self.compute_score.f1(item['emotion'],pred_emotion)
                    gender_f1+=self.compute_score.f1(item['gender'],pred_gender)

            age_acc/=len(valid)
            race_acc/=len(valid)
            masked_acc/=len(valid)
            skintone_acc/=len(valid)
            emotion_acc/=len(valid)
            gender_acc/=len(valid)

            age_f1/=len(valid)
            race_f1/=len(valid)
            masked_f1/=len(valid)
            skintone_f1/=len(valid)
            emotion_f1/=len(valid)
            gender_f1/=len(valid)

            avg_valid_acc = (age_acc+race_acc+masked_acc+skintone_acc+emotion_acc+gender_acc)/6.
            avg_valid_f1 = (age_f1+race_f1+masked_f1+skintone_f1+emotion_f1+gender_f1)/6.

            print(f"epoch {epoch + 1}/{self.num_epochs + initial_epoch}")
            print(f"train loss: {train_loss:.4f}")
            print(f"Age Acc: {age_acc:.4f}, Race Acc: {race_acc:.4f}, Masked Acc: {masked_acc:.4f}, Skintone Acc: {skintone_acc:.4f}, Emotion Acc: {emotion_acc:.4f}, Gender Acc: {gender_acc:.4f}")
            print(f"Age F1: {age_f1:.4f}, Race F1: {race_f1:.4f}, Masked F1: {masked_f1:.4f}, Skintone F1: {skintone_f1:.4f}, Emotion F1: {emotion_f1:.4f}, Gender F1: {gender_f1:.4f}")
            print(f"Avg Acc: {avg_valid_acc:.4f}")
            print(f"Avg F1: {avg_valid_f1:.4f}")


            with open('log.txt', 'a') as file:
                file.write(f"epoch {epoch + 1}/{self.num_epochs + initial_epoch}\n")
                file.write(f"train loss: {train_loss:.4f}\n")
                file.write(f"Age Acc: {age_acc:.4f}, Race Acc: {race_acc:.4f}, Masked Acc: {masked_acc:.4f}, Skintone Acc: {skintone_acc:.4f}, Emotion Acc: {emotion_acc:.4f}, Gender Acc: {gender_acc:.4f}\n")
                file.write(f"Age F1: {age_f1:.4f}, Race F1: {race_f1:.4f}, Masked F1: {masked_f1:.4f}, Skintone F1: {skintone_f1:.4f}, Emotion F1: {emotion_f1:.4f}, Gender F1: {gender_f1:.4f}\n")
                file.write(f"Avg Acc: {avg_valid_acc:.4f}\n")
                file.write(f"Avg F1: {avg_valid_f1:.4f}\n")

            if self.best_metric =='accuracy':
                score=avg_valid_acc
            if self.best_metric=='f1':
                score=avg_valid_f1
            # save the last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'self.optimizer_state_dict': self.optimizer.state_dict(),
                'score': score}, os.path.join(self.save_path, 'last_model.pth'))

            # save the best model
            if epoch > 0 and score <= best_score:
                threshold += 1
            else:
                threshold = 0

            if score > best_score:
                best_score = score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.base_model.state_dict(),
#                     'self.optimizer_state_dict': self.optimizer.state_dict(),
                    'score':score}, os.path.join(self.save_path, 'best_model.pth'))
                print(f"saved the best model with {self.best_metric} of {score:.4f}")

            # early stopping
            if threshold >= self.patience:
                print(f"early stopping after epoch {epoch + 1}")
                break
