# dataset.py

import cv2
import numpy as np
import pandas as pd
from config import DIR_INPUT, SIZE
from albumentations import Compose, Resize, HorizontalFlip, VerticalFlip, RandomRotate90, Normalize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

transform = {
    'train' : Compose([
        Resize(SIZE[0], SIZE[1], always_apply=True),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Normalize(p=1.0),
        ToTensorV2(p=1.0)
    ]),
    'valid': Compose([
        Resize(SIZE[0], SIZE[1], always_apply=True),
        Normalize(p=1.0),
        ToTensorV2(p=1.0)
    ]),
    'test_tta': Compose([
        Resize(SIZE[0], SIZE[1], always_apply=True),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Normalize(p=1.0),
        ToTensorV2(p=1.0)
    ])
}

class PLANT(Dataset):
    def __init__(self, df, transform=None, train=True):
        self.df = df
        self.transform = transform
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path = self.df.iloc[idx]['image_id']
        image = cv2.imread(DIR_INPUT + f"/images/{path}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        if self.train:
            label = np.argmax(self.df.iloc[idx][1:].values).astype(int)
            return {'image': image, 'label': label}
        return {'image': image}
