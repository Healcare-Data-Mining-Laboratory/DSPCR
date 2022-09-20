import re
import torch
from torch import nn
from torch.nn import functional as F
import math
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
from transformers import BartModel,BartTokenizer
import os
from collections import deque
import torch.optim as optim
import sys,logging
from transformers import AutoTokenizer, AutoModel
from numpy.testing import assert_almost_equal
import botorch


class mllt(nn.Module):
    def __init__(self,class_3,latent_ndims):
        super(mllt, self).__init__()
        self.class_3 = class_3
        self.latent_ndims = latent_ndims
        self.hidden_size = 768
        self.alpha = 5
        self.drop_out = nn.Dropout(0.3)
        self.fc_key = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.fc_query = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.fc_value = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.cluster_layer = nn.Sequential(
            nn.Linear(self.hidden_size//2, self.hidden_size//2),
            nn.PReLU()
            )
        self.MLPs = nn.Sequential(
                    nn.Linear(self.hidden_size//2, 100),
                    nn.Dropout(0.3),
                    nn.Linear(100, 3),
                    )
        # self.MLPs = nn.Sequential(
        #     nn.Linear(self.hidden_size//2, 3),
        #     )
        self.encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def get_cluster_prob(self, embeddings,Center):
        # print(embeddings.unsqueeze(1)[[0],:,:] - Center)

        norm_squared = torch.sum((embeddings.unsqueeze(1) - Center) ** 2, -1)

        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        # print(numerator.shape)
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def target_distribution(self,batch):
        weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
        return (weight.t() / torch.sum(weight, 1)).t()


    def clustering(self,Ztd,Center):
        Center = self.cluster_layer(Center) ## 8x384

        phi = self.get_cluster_prob(Ztd,Center)#phi batch x 10 x 1
        # print("phi: ",phi[:,:])
        # print("z",Ztd[:3,:10].unsqueeze(1))
        # print("Center",Center[:5,:10])
        # print("divition: ",(Ztd[:5,:].unsqueeze(1) - Center).sum(-1)[:,:10])
        
        cluster_target =self.target_distribution(phi).detach()
        # print(phi[:1,:],cluster_target[:1,:])

        ct = self.drop_out(Center * phi.unsqueeze(-1)).sum(1)
        return ct,phi,Center,cluster_target
    def cross_attention(self,v,c):
       
        B, Nt, E = v.shape
        v = v / math.sqrt(E)
        # c = c / math.sqrt(E)
        # print("max v",torch.max(self.sigmoid(v)))
        # print("max c",torch.max(self.sigmoid(c)))
        v = self.drop_out(self.fc_key(v))
        c = self.drop_out(self.fc_query(c))
        # v = self.drop_out(self.sigmoid(self.fc_key(v)))
        # c = self.drop_out(self.sigmoid(self.fc_query(c)))
        g = torch.bmm(v, c.transpose(-2, -1))
        # print("max v",torch.max(v))
        # print("max c",torch.max(c))

        m = F.max_pool2d(g,kernel_size = (1,g.shape[-1])).squeeze(1)  # [b, l, 1]

        b = torch.softmax(m, dim=1)  # [b, l, 1]
        # print("b: ",b.squeeze().squeeze(),torch.max(b),torch.min(b))

        return b  
    
    def approximation(self,Ot,flag,self_att):
              
        Ot_E_batch = self.encoder(**Ot).last_hidden_state

        if self_att == "self_att":
            attention_weights =  self.cross_attention(Ot_E_batch,Ot_E_batch)
            Ot_E_batch = self.drop_out(self.fc_value(Ot_E_batch))
            Ot_E_batch_cross_attention =    (Ot_E_batch * attention_weights).sum(1)
      
        else:
            Ot_E_batch_cross_attention = self.drop_out(self.fc_value(Ot_E_batch)).mean(1)

        return Ot_E_batch_cross_attention
   

    def forward(self,Ot,Center,flag,self_att):
        Ztd = self.approximation(Ot,flag,self_att)
        u,phi,Center,cluster_target = self.clustering(Ztd,Center)
        y =  self.sigmoid(self.MLPs(u))
        return y,phi,cluster_target



        

       

    


   