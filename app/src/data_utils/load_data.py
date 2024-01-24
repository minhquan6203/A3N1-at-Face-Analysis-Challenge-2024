from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Optional
import pandas as pd
import os
import numpy as np
class CustomDataset(Dataset):
    def __init__(self, data, image_folder, with_labels=True):
        self.data = data  # pandas dataframe
        self.with_labels = with_labels
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name=os.path.join(self.image_folder, self.data.loc[index, 'crop_name'])
        if self.with_labels:  # True if the dataset has labels
            age=self.data.loc[index, 'age']
            race=self.data.loc[index, 'race']
            masked=self.data.loc[index, 'masked']
            skintone=self.data.loc[index, 'skintone']
            emotion=self.data.loc[index, 'emotion']
            gender=self.data.loc[index, 'gender']
            item={'image_name':img_name,
                  'age':age,
                  'race':race,
                  'masked':masked,
                  'skintone': skintone,
                  'emotion':emotion,
                  'gender': gender}
        else:
            item={'image_name':img_name}
        return item

class Get_Loader:
    def __init__(self, config):
        self.train_path=config['data']['train_dataset']
        self.image_folder_train_dev = config["data"]["images_folder"]
        self.train_batch=config['train']['per_device_train_batch_size']

        self.val_path=config['data']['val_dataset']
        self.val_batch=config['train']['per_device_valid_batch_size']

        self.test_path=config['infer']['test_dataset']
        self.image_folder_test = config["infer"]["images_folder"]
        self.test_batch=config['infer']['batch_size']

    def load_train_dev(self):
        print("Reading training data...")
        train_df=pd.read_csv(self.train_path)
        age_space = sorted(list(np.unique(train_df['age'])))
        age_to_index = {label: index for index, label in enumerate(age_space)}
        train_df['age'] = train_df['age'].map(age_to_index)

        race_space = sorted(list(np.unique(train_df['race'])))
        race_to_index = {label: index for index, label in enumerate(race_space)}
        train_df['race'] = train_df['race'].map(race_to_index)

        masked_space = sorted(list(np.unique(train_df['masked'])))
        masked_to_index = {label: index for index, label in enumerate(masked_space)}
        train_df['masked'] = train_df['masked'].map(masked_to_index)

        skintone_space = sorted(list(np.unique(train_df['skintone'])))
        skintone_to_index = {label: index for index, label in enumerate(skintone_space)}
        train_df['skintone'] = train_df['skintone'].map(skintone_to_index)

        emotion_space = sorted(list(np.unique(train_df['emotion'])))
        emotion_to_index = {label: index for index, label in enumerate(emotion_space)}
        train_df['emotion'] = train_df['emotion'].map(emotion_to_index)

        gender_space = sorted(list(np.unique(train_df['gender'])))
        gender_to_index = {label: index for index, label in enumerate(gender_space)}
        train_df['gender'] = train_df['gender'].map(gender_to_index)
        train_set = CustomDataset(data=train_df,image_folder=self.image_folder_train_dev)

        print("Reading validation data...")
        val_df=pd.read_csv(self.val_path)
        val_df['age'] = val_df['age'].map(age_to_index)
        val_df['race'] = val_df['race'].map(race_to_index)
        val_df['masked'] = val_df['masked'].map(masked_to_index)
        val_df['skintone'] = val_df['skintone'].map(skintone_to_index)
        val_df['emotion'] = val_df['emotion'].map(emotion_to_index)
        val_df['gender'] = val_df['gender'].map(gender_to_index)
        val_set = CustomDataset(data=val_df,image_folder=self.image_folder_train_dev)
        train_loader = DataLoader(train_set, batch_size=self.train_batch, num_workers=2,shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.val_batch, num_workers=2,shuffle=True)
        return train_loader, val_loader

    def load_test(self):
        test_df=pd.read_csv(self.test_path)
        print("Reading testing data...")
        test_set = CustomDataset(data=test_df,image_folder=self.image_folder_test,with_labels=False)
        test_loader = DataLoader(test_set, batch_size=self.test_batch, num_workers=2, shuffle=False)
        return test_loader

def create_ans_space(config: Dict):
    train_path=config['data']['train_dataset']
    train_df=pd.read_csv(train_path)
    age_space = sorted(list(np.unique(train_df['age'])))
    race_space = sorted(list(np.unique(train_df['race'])))
    masked_space = sorted(list(np.unique(train_df['masked'])))
    skintone_space = sorted(list(np.unique(train_df['skintone'])))
    emotion_space = sorted(list(np.unique(train_df['emotion'])))
    gender_space = sorted(list(np.unique(train_df['gender'])))
    return age_space, race_space, masked_space, skintone_space, emotion_space, gender_space