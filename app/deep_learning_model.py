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
        "image_encoder": "WinKawaks/vit-small-patch16-224", #WinKawaks/vit-small-patch16-224, google/vit-base-patch16-224, google/vit-large-patch16-224
        "freeze": False,
        "d_features":  384, #ResNet 2048, VIT base 768
        "d_model":   384, #small 384, base 512, large 768
        "dropout": 0.2
    },
    "svm":{
        "gamma": 0.1,
        "degree": 2,
        "r": 1,
        "kernel_type": "custom"
    },
    "model": {
        "type_model": "svm",
        "intermediate_dims": 384,
        "dropout": 0.2
    }
}

class Face_Analysis:
    def __init__(self, project="app/static/", name="", exist_ok=True):
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.Yolomodel = YOLO('checkpoint/yolo_m_best.pt', task="detect")
        self.Yolomodel.to(self.device)
        self.Yolomodel.predict('app/static/logo.png', save=False, save_crop=False, verbose=False)
        
        self.classify_model = build_model(config)
        self.classify_model.to(self.device)
        self.checkpoint = torch.load('checkpoint/best_model_custom_svm_vit_small.pth',map_location=self.device)
        self.classify_model.load_state_dict(self.checkpoint['model_state_dict'])
        self.age_space, self.race_space, self.masked_space, self.skintone_space, self.emotion_space, self.gender_space=create_ans_space(config)
                  
    def perform_yolo_detection(self, image_path, save=True, save_crop=False):
        try:
            results = self.Yolomodel.predict(image_path, save=save, save_crop=save_crop,
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
