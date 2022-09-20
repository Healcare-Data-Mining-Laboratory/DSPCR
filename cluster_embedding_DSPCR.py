import re
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_packed import PatientDataset
import torch.nn.utils.rnn as rnn_utils
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm
import numpy as np
import os
from collections import deque
import torch.optim as optim
import sys,logging
   
class mllt(nn.Module):
    def __init__(self):
        super(mllt, self).__init__()
        self.hidden_size = 768
        self.encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.fc_key = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.fc_query = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.fc_value = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.fuse_fc = nn.Linear(self.hidden_size,self.hidden_size//2)

        self.drop_out = nn.Dropout(0.3)
   

    def cross_attention(self,v,c):
        B, Nt, E = v.shape
        v = v / math.sqrt(E)
        # print("v :", v)
        # print("c :", c)

        v = self.drop_out(self.fc_key(v))
        c = self.drop_out(self.fc_query(c))
        g = torch.bmm(v, c.transpose(-2, -1))

        m = F.max_pool2d(g,kernel_size = (1,g.shape[-1])).squeeze(1)  # [b, l, 1]

        b = torch.softmax(m, dim=1)  # [b, l, 1]
        # print("b: ",b[[1],:,:].squeeze().squeeze())
        return b   


    def approximation(self, Ot,self_att):
        Ot_E_batch = self.encoder(**Ot).last_hidden_state
        if self_att == "self_att":
            attention_weights =  self.cross_attention(Ot_E_batch,Ot_E_batch)
            Ot_E_batch = self.drop_out(self.fc_value(Ot_E_batch))
            Ztd =   (Ot_E_batch * attention_weights).sum(1)
   
        else:
            return self.drop_out(self.fc_value(Ot_E_batch)).mean(1) 
 



    def forward(self,Ot,label_token,self_att):
        return self.approximation(Ot,label_token,self_att)
           

   



