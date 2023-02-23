import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from timm import create_model
import torch

class BreastCancerDetectionModel(BaseModel):
    def __init__(self, cols_num_classes, model_name, dropout=0.):
        '''
        cols_num_classes: list of number of classes in each columnn
        '''
        super().__init__()
        self.model = create_model(model_name, pretrained=True, num_classes=0, drop_rate=dropout)
        self.backbone_output_dim = self.model(torch.randn(1, 3, 512, 512)).shape[-1]
        
        # Head Cancer probabiltiy
        self.cancer_head = nn.Linear(self.backbone_output_dim, 1)
        self.cols_values_head = torch.nn.ModuleList(torch.nn.Linear(self.backbone_output_dim, n) for n in cols_num_classes)
    def forward(self, x):
        '''
        x: img (batchh_size, 3, 512, 512))
        
        RETURN: 
        cancer_pred: tensor (batch_size, 1)
        cols_values_pred: LIST of tensors (batch_size, n_classes) for n_classes is number of classes in each column
        '''
        x = self.model(x)
        cancer_pred = self.cancer_head(x).squeeze() # torch size []
        cols_values_pred = []
        for col_head in self.cols_values_head:
            cols_values_pred.append(col_head(x).squeeze())
        return cancer_pred, cols_values_pred
    

    def predict(self, x): 
        cancer_pred, cols_values_preds = self.forward(x)
        softmax_cols_values_preds = []
        for a in cols_values_preds:
            softmax_cols_values_preds.append(torch.softmax(a, dim=-1))
        return torch.sigmoid(cancer_pred), softmax_cols_values_preds
