# model.py

import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torch.cuda import device

class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super(PlantModel, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b5')
        self.base_model._fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    
    def forward(self, x):
        return self.base_model(x)
