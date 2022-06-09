import re
import torch
from torch import nn
from torch.nn import functional as F
import math
import torch.nn.utils.rnn as rnn_utils
from transformers import AutoTokenizer, AutoModel
from torch.distributions.studentT import StudentT
from tqdm import tqdm
import numpy as np
import os
from collections import deque
import torch.optim as optim
import sys,logging

class mllt(nn.Module):
    def __init__(self,class_3):
        super(mllt, self).__init__()
        self.hidden_size = 768
        self.encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.fc_key = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.fc_query = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.fc_value = nn.Linear(self.hidden_size,self.hidden_size//2)

        self.cluster_layer = nn.Sequential(
            nn.Linear(self.hidden_size//2, self.hidden_size//2),
            nn.PReLU()
            )
        self.alpha = 5

        if class_3:
            self.MLPs = nn.Sequential(
                nn.Linear(self.hidden_size//2, 3),
                )
        else:
            self.MLPs = nn.Sequential(
                nn.Linear(self.hidden_size//2, 25),
                )
            
        self.drop_out = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
   



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
            Ztd =   self.drop_out(Ot_E_batch * attention_weights).sum(1)
            return Ztd        
 
    def get_cluster_prob(self, embeddings,Center):

        norm_squared = torch.sum((embeddings.unsqueeze(1) - Center) ** 2, -1)

        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def target_distribution(self,batch):
        weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
        return (weight.t() / torch.sum(weight, 1)).t()
    def selector(self,Center,phi):
        index_list = torch.max(phi,dim = -1).indices
        
        tmp = []
        # print("st: ",index_list)

        for i in index_list:
            selected_tensor = Center[[i],:]
            tmp.append(selected_tensor)
        return torch.cat(tmp,0)

    def clustering(self,Ztd,Center):
        Center = self.cluster_layer(Center) ## 8x384

        phi = self.get_cluster_prob(Ztd,Center)#phi batch x 10 x 1

        
        cluster_target =self.target_distribution(phi).detach()
        # print(phi[:1,:],cluster_target[:1,:])

        ct = self.drop_out(Center * phi.unsqueeze(-1)).sum(1)
        return ct,phi,Center,cluster_target

    def forward(self,Ot,Center,self_att,train_base):


        Ztd = self.approximation(Ot,self_att)
        if train_base:
            Yt =  self.sigmoid(self.MLPs(Ztd))
            # print("Yt",Yt)
            return Yt

        else:
            ct,phi,Center,cluster_target = self.clustering(Ztd,Center)
            Yt =  self.sigmoid(self.MLPs(ct))
            return Yt,Center,phi,cluster_target

   