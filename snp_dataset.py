# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 11:23:07 2022

@author: Niels
"""
import torch
from torch.utils.data import Dataset

class snp_Dataset(Dataset):
    def __init__(self,data,labels):
        super(snp_Dataset).__init__()
        self.data = torch.tensor(data,dtype = torch.float32)
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        genotype = self.data[idx]
        label = self.labels[idx]
        return genotype, label