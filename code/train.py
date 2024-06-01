# train.py
import warnings  
warnings.filterwarnings('ignore')
import torch
import numpy as np
from config import *
from dataset import PLANT, transform
from model import PlantModel
from torch.utils.data import DataLoader
from torch.optim import Adam
import pandas as pd
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score
from utils import seed_everything, display_training_curves
import os
import cv2
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,precision_score,recall_score,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from transformers import get_cosine_schedule_with_warmup
from albumentations import *
from albumentations.pytorch import ToTensorV2

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from efficientnet_pytorch import EfficientNet

def train_epochs(model, criterion, optimizer,scheduler, dataloader_train, dataloader_valid):
    global results
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch,EPOCHS-1))
        since = time.time()
        model.train()
        training_accuracy  = []
        training_loss = []
        for bi, d in enumerate(tqdm(dataloader_train, total=int(len(dataloader_train)))):
            inputs = d["image"]
            labels = d["label"]
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                labels = labels.squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                outputs = torch.max(outputs,1)[1]
                outputs = outputs.cpu().detach().numpy()
                labels = labels.cpu().numpy()
                training_accuracy.append(accuracy_score(outputs,labels))
                training_loss.append(loss.item())
        print('Training accuracy: {:.4f} and Training Loss: {:.4f}'.format(np.mean(training_accuracy),np.mean(training_loss)))

        
                                     
        model.eval()
        validation_loss = []
        validation_labels = []
        validation_outputs = []
        with torch.no_grad():
            for bi,d in enumerate(tqdm(dataloader_valid,total=int(len(dataloader_valid)))):
                inputs = d["image"]
                labels = d["label"]
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
                outputs = model(inputs)
                labels = labels.squeeze()
                loss = criterion(outputs,labels)
                outputs_softmax = F.softmax(outputs).cpu().detach().numpy()
                labels_onehot = torch.eye(4).cuda()[labels].cpu().numpy()
                validation_labels.extend(labels_onehot)
                validation_outputs.extend(outputs_softmax)
                validation_loss.append(loss.item())
        precision = precision_score(np.argmax(validation_labels,axis=1),np.argmax(validation_outputs,axis=1),average='macro')
        recall = recall_score(np.argmax(validation_labels,axis=1),np.argmax(validation_outputs,axis=1),average='macro')
        accuracy = accuracy_score(np.argmax(validation_labels,axis=1),np.argmax(validation_outputs,axis=1))
        roc = roc_auc_score(validation_labels,validation_outputs,average='macro')
        print('Validation accuracy: {:.4f} and Validation Loss: {:.4f} and roc_auc_score: {:.4f}'.format(accuracy,\
                            np.mean(validation_loss),roc))
        res = pd.DataFrame([[np.mean(training_loss),np.mean(training_accuracy),np.mean(validation_loss),\
                             accuracy,precision,recall,roc]],columns=results.columns)
        results = pd.concat([results,res])
        scheduler.step()
    return results.iloc[-1]


def main():
    seed_everything(SEED)  
    global device
    device = torch.device(DEVICE)
    train_df = pd.read_csv(DIR_INPUT + '/train.csv')
    test_df = pd.read_csv(DIR_INPUT + '/test.csv')
    cols = list(train_df.columns[1:])
    train, valid = train_test_split(train_df,test_size = 0.2,random_state = SEED)

    dataset_train = PLANT(df=train, transform=transform['train'])
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, drop_last=True)

    dataset_valid = PLANT(df=valid, transform=transform['valid'])
    dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, drop_last=True)

    dataset_test = PLANT(test_df,transform=transform['valid'],train=False)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

    dataset_test_tta = PLANT(test_df,transform=transform['test_tta'],train=False)
    dataloader_test_tta = DataLoader(dataset_test_tta, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

    
    new_model = PlantModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(new_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE,gamma=0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP, num_training_steps=EPOCHS)
    global results
    results = pd.DataFrame(columns=['training_loss','training_accuracy','validation_loss','validation_accuracy','precision','recall','roc_auc_score'])

    train_epochs(new_model, criterion, optimizer,scheduler, dataloader_train, dataloader_valid)

    results.reset_index(drop=True,inplace=True)
    display_training_curves(results['training_loss'], results['validation_loss'], 'loss', 311, 'images/loss.png')
    display_training_curves(results['training_accuracy'], results['validation_accuracy'], 'accuracy', 312, 'images/acc.png')
    display_training_curves(1, results['roc_auc_score'], 'roc_auc_score', 313, 'images/auc.png')

    display_training_curves(1, results['precision'], 'precision', 211, 'images/precision.png')
    display_training_curves(1, results['recall'], 'recall', 212, 'images/recall.png')

    new_model.eval()
    test_pred = np.zeros((len(test_df),4))
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader_test,total=int(len(dataloader_test)))):
            inputs = data['image']
            inputs = inputs.to(device, dtype=torch.float)
            predict = new_model(inputs)
            test_pred[i*len(predict):(i+1)*len(predict)] = predict.detach().cpu().squeeze().numpy()
            
            
    submission_df = pd.read_csv(DIR_INPUT + '/sample_submission.csv')
    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = F.softmax(torch.from_numpy(test_pred),dim=1)
    submission_df.to_csv('submission.csv', index=False)
    pd.Series(np.argmax(submission_df[cols].values,axis=1)).value_counts()

    new_model.eval()
    test_pred = np.zeros((len(test_df),4))
    for i in range(TTA):
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader_test_tta,total=int(len(dataloader_test_tta)))):
                inputs = data['image']
                inputs = inputs.to(device, dtype=torch.float)
                predict = new_model(inputs)
                test_pred[i*len(predict):(i+1)*len(predict)] += predict.detach().cpu().squeeze().numpy()
                
                
    submission_df = pd.read_csv(DIR_INPUT + '/sample_submission.csv')
    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = F.softmax(torch.from_numpy(test_pred/TTA),dim=1)
    submission_df.to_csv('submission_tta.csv', index=False)
    pd.Series(np.argmax(submission_df[cols].values,axis=1)).value_counts()
    
if __name__ == '__main__':
    main()