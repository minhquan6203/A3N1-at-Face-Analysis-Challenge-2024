from ultralytics import YOLO
from PIL import Image
from IPython.display import display
import torch
from src.data_utils import create_ans_space
from src.model import build_model
import os

config= {
    "data": {
        "train_dataset": "app/data/train.csv",
    },
    "vision_embedding": {
        "type": "VIT", #ResNet or VIT
        "image_encoder": "google/vit-base-patch16-224-in21k",
        "freeze": False,
        "d_features":  768, #ResNet 2048, VIT base 768
        "d_model":   512,
        "dropout": 0.2
    },
    "attention": {
        "heads": 8,
        "d_model": 512,
        "d_key": 64,
        "d_value": 64,
        "d_ff": 2048,
        "d_feature": 2048,
        "dropout": 0.2,
    },
    "encoder": {
        "d_model": 512,
        "layers": 4
    },
    "model": {
        "type_model": "trans",
        "intermediate_dims": 512,
        "dropout": 0.2
    },
    "train": {
        "output_dir": "app/best_model.pth",
        "seed": 12345,
        "num_train_epochs": 20,
        "patience": 4,
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-4,
        "metric_for_best_model": "accuracy",
        "per_device_train_batch_size": 16,
        "per_device_valid_batch_size": 16,
    },
}

class Face_Analysis:
    def __init__(self, project="app/static/", name="", exist_ok=True):
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs('checkpoint',exist_ok=True) 
        if not os.path.exists('checkpoint/best.pt'):
            os.system("curl -L -o 'checkpoint/best.pt' 'https://drive.usercontent.google.com/download?id=1ttBF9wdPK6yqtrl_yiZL-Ly7n0T5wJTF&export=download&authuser=1&confirm=t'")
        
        if not os.path.exists('checkpoint/best_model.pth'):
            os.system("curl -L -o 'checkpoint/best_model.pth' 'https://drive.usercontent.google.com/download?id=1gA4r3MBvjmr9JXG7R3ACkwVWKeFKQaVS&export=download&authuser=1&confirm=t'")
        
        self.Yolomodel = YOLO('checkpoint/best.pt').to(self.device)
        self.classify_model = build_model(config)
        self.classify_model.to(self.device)
        self.checkpoint = torch.load('checkpoint/best_model.pth',map_location=self.device)
        self.classify_model.load_state_dict(self.checkpoint['model_state_dict'])
        self.age_space, self.race_space, self.masked_space, self.skintone_space, self.emotion_space, self.gender_space=create_ans_space(config)
                  
    def perform_yolo_detection(self, image_path, save=True, save_crop=False):
        try:
            
            results = self.Yolomodel(image_path, save=save, save_crop=save_crop,
                                project=self.project, name=self.name, exist_ok=self.exist_ok)
            return results
        except Exception as e:
            raise Exception(f"detection failed: {str(e)}")


    def perform_face_classify(self, croped_image_name):
        try:
            with torch.no_grad():
                logits = self.classify_model(croped_image_name)
                age = self.age_space[logits['head_age'].argmax(-1)]
                race = self.race_space[logits['head_race'].argmax(-1)]
                masked = self.masked_space[logits['head_masked'].argmax(-1)]
                skintone = self.skintone_space[logits['head_skintone'].argmax(-1)]
                emotion = self.emotion_space[logits['head_emotion'].argmax(-1)]
                gender = self.gender_space[logits['head_gender'].argmax(-1)]
                extracted_text=f"age: {age}, race: {race}, masked: {masked}, skintone: {skintone}, emotion: {emotion}, gender: {gender}"
                return extracted_text
        except Exception as e:
            raise Exception(f"failed: {str(e)}")
